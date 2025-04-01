from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from vit.vit_triplane import Triplane
from nsr.triplane import TriplaneMesh
import dnnlib
from guided_diffusion import dist_util, logger
from ipdb import set_trace as st
import vit.vision_transformer as vits
from .confnet import ConfNet

from ldm.modules.diffusionmodules.model import (
    Encoder, MVEncoder, MVEncoderGS, MVEncoderGSDynamicInp
)
from ldm.modules.diffusionmodules.mv_unet import MVUNet, LGM_MVEncoder


class AE(torch.nn.Module):
    """Auto-Encoder model that combines encoder and decoder for 3D object representation.
    
    The encoder extracts features from input multiview images, while the decoder generates 
    3D triplane representations and renders novel views.
    """

    def __init__(self,
                 encoder,
                 decoder, 
                 img_size,
                 encoder_cls_token,
                 decoder_cls_token,
                 preprocess,
                 use_clip,
                 dino_version='v1',
                 clip_dtype=None,
                 no_dim_up_mlp=False,
                 dim_up_mlp_as_func=False,
                 uvit_skip_encoder=False,
                 confnet=None) -> None:
        super().__init__()
        
        # Core components
        self.encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.preprocess = preprocess
        
        # Model configuration
        self.encoder_cls_token = encoder_cls_token
        self.decoder_cls_token = decoder_cls_token
        self.use_clip = use_clip
        self.dino_version = dino_version
        self.confnet = confnet
        self.dim_up_mlp = None
        self.dim_up_mlp_as_func = dim_up_mlp_as_func

        # Handle DINO v2 specific setup
        if self.dino_version == 'v2':
            self.encoder.mask_token = None
            self.decoder.vit_decoder.mask_token = None

        # Configure non-SD models
        if 'sd' not in self.dino_version:
            self.uvit_skip_encoder = uvit_skip_encoder
            if uvit_skip_encoder:
                logger.log(f'Enabling UViT with {len(self.encoder.blocks)} encoder blocks')
                # Add skip connections to second half of blocks
                for blk in self.encoder.blocks[len(self.encoder.blocks) // 2:]:
                    blk.skip_linear = nn.Linear(2 * self.encoder.embed_dim,
                                              self.encoder.embed_dim)
                    # Initialize skip connection weights
                    nn.init.constant_(blk.skip_linear.weight, 0)
                    if isinstance(blk.skip_linear, nn.Linear) and blk.skip_linear.bias is not None:
                        nn.init.constant_(blk.skip_linear.bias, 0)
            else:
                logger.log('UViT disabled')
        else:
            # Configure SD models
            if 'dit' not in self.dino_version:
                # Clean up unused components for DINO ViT
                self.decoder.vit_decoder.cls_token = None 
                self.decoder.vit_decoder.patch_embed.proj = nn.Identity()
                self.decoder.triplane_decoder.planes = None
                self.decoder.vit_decoder.mask_token = None

            if self.use_clip:
                self.clip_dtype = clip_dtype
            elif not no_dim_up_mlp and self.encoder.embed_dim != self.decoder.vit_decoder.embed_dim:
                # Add dimension upsampling if needed
                self.dim_up_mlp = nn.Linear(
                    self.encoder.embed_dim,
                    self.decoder.vit_decoder.embed_dim)

        torch.cuda.empty_cache()

    def encode(self, *args, **kwargs):
        """Encode input images into latent representations"""
        if not self.use_clip:
            if self.dino_version == 'v1':
                latent = self.encode_dinov1(*args, **kwargs)
            elif self.dino_version == 'v2':
                latent = self.encode_dinov2_uvit(*args, **kwargs) if self.uvit_skip_encoder else self.encode_dinov2(*args, **kwargs)
            else:
                latent = self.encoder(*args)
        else:
            latent = self.encode_clip(*args, **kwargs)
        return latent

    def encode_dinov1(self, x):
        """DINO v1 encoding process"""
        x = self.encoder.prepare_tokens(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return x[:, 1:] if not self.encoder_cls_token else x

    def encode_dinov2(self, x):
        """DINO v2 encoding process"""
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)
        for blk in self.encoder.blocks:
            x = blk(x)
        x_norm = self.encoder.norm(x)
        return x_norm[:, 1:] if not self.encoder_cls_token else x_norm

    def encode_dinov2_uvit(self, x):
        """DINO v2 encoding with UViT skip connections"""
        x = self.encoder.prepare_tokens_with_masks(x, masks=None)
        skips = [x]
        
        # First half of blocks
        for blk in self.encoder.blocks[:len(self.encoder.blocks)//2 - 1]:
            x = blk(x)
            skips.append(x)
            
        # Middle blocks
        for blk in self.encoder.blocks[len(self.encoder.blocks)//2 - 1:len(self.encoder.blocks)//2]:
            x = blk(x)
            
        # Second half with skip connections
        for blk in self.encoder.blocks[len(self.encoder.blocks)//2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()], dim=-1))
            x = blk(x)
            
        x_norm = self.encoder.norm(x)
        return x_norm[:, 1:] if not self.decoder_cls_token else x_norm

    def encode_clip(self, x):
        """CLIP model encoding process"""
        x = self.encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)
        return self.encoder.ln_post(x[:, 1:, :])

    def decode_wo_triplane(self, latent, c=None, img_size=None):
        """Decode latent without triplane rendering"""
        if img_size is None:
            img_size = self.img_size

        if self.dim_up_mlp is not None:
            if not self.dim_up_mlp_as_func:
                latent = self.dim_up_mlp(latent)
            else:
                return self.decoder.vit_decode(latent, img_size, dim_up_mlp=self.dim_up_mlp)

        return self.decoder.vit_decode(latent, img_size, c=c)

    def decode(self, latent, c, img_size=None, return_raw_only=False):
        """Full decoding process including triplane rendering"""
        latent = self.decode_wo_triplane(latent, img_size=img_size, c=c)
        return self.decoder.triplane_decode(latent, c)

    def decode_after_vae_no_render(self, ret_dict, img_size=None):
        """Decode after VAE without rendering"""
        if img_size is None:
            img_size = self.img_size

        assert self.dim_up_mlp is None
        latent = self.decoder.vit_decode_backbone(ret_dict, img_size)
        ret_dict = self.decoder.vit_decode_postprocess(latent, ret_dict)
        return ret_dict

    def decode_after_vae(self, ret_dict, c, img_size=None, return_raw_only=False):
        """Full decoding process after VAE"""
        ret_dict = self.decode_after_vae_no_render(ret_dict, img_size)
        return self.decoder.triplane_decode(ret_dict, c)

    def decode_confmap(self, img):
        """Generate confidence map from input image"""
        assert self.confnet is not None
        return self.confnet(img)  # Bx1xHxW

    def encode_decode(self, img, c, return_raw_only=False):
        """Full encode-decode pipeline"""
        latent = self.encode(img)
        pred = self.decode(latent, c, return_raw_only=return_raw_only)
        if self.confnet is not None:
            pred.update({'conf_sigma': self.decode_confmap(img)})
        return pred

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        """Convert input image to index representation"""
        ret_dict = self.decoder.vae_reparameterization(self.encode(inp_img_no_grad), None)
        f = self.decoder.superresolution['quant_conv'](ret_dict['latent_normalized_2Ddiffusion'])
        return self.decoder.superresolution['quantize'].f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def forward(self, img=None, c=None, latent=None, behaviour='enc_dec', coordinates=None, directions=None, return_raw_only=False, *args, **kwargs):
        """Forward pass with multiple behavior modes"""
        if behaviour == 'enc_dec':
            return self.encode_decode(img, c, return_raw_only=return_raw_only)

        elif behaviour == 'enc':
            return self.encode(img)

        elif behaviour == 'dec':
            assert latent is not None
            return self.decode(latent, c, self.img_size, return_raw_only=return_raw_only)

        elif behaviour == 'dec_wo_triplane':
            assert latent is not None
            return self.decode_wo_triplane(latent, self.img_size)

        elif behaviour == 'enc_dec_wo_triplane':
            latent = self.encode(img)
            return self.decode_wo_triplane(latent, img_size=self.img_size, c=c)

        elif behaviour == 'encoder_vae':
            latent = self.encode(img)
            return self.decoder.vae_reparameterization(latent, True)

        elif behaviour == 'decode_after_vae_no_render':
            return self.decode_after_vae_no_render(latent, self.img_size)

        elif behaviour == 'decode_after_vae':
            return self.decode_after_vae(latent, c, self.img_size)

        elif behaviour == 'triplane_dec':
            assert latent is not None
            return self.decoder.triplane_decode(latent, c, return_raw_only=return_raw_only, **kwargs)

        elif behaviour == 'triplane_decode_grid':
            assert latent is not None
            return self.decoder.triplane_decode_grid(latent, **kwargs)

        elif behaviour == 'vit_postprocess_triplane_dec':
            assert latent is not None
            latent = self.decoder.vit_decode_postprocess(latent)
            return self.decoder.triplane_decode(latent, c)

        elif behaviour == 'triplane_renderer':
            assert latent is not None
            return self.decoder.triplane_renderer(latent, coordinates, directions)

        elif behaviour == 'get_rendering_kwargs':
            return self.decoder.triplane_decoder.rendering_kwargs


def create_3DAE_model(
        arch_encoder,
        arch_decoder,
        dino_version='v1',
        img_size=[224],
        patch_size=16,
        in_chans=384,
        num_classes=0,
        embed_dim=1024,  # Check ViT encoder dim
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        out_chans=96,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=32,
        encoder_cls_token=False,
        decoder_cls_token=False,
        c_dim=25,  # Conditioning label (C) dimensionality.
        image_size=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        rendering_kwargs={},
        load_pretrain_encoder=False,
        decomposed=True,
        triplane_size=224,
        ae_classname='ViTTriplaneDecomposed',
        use_clip=False,
        sr_kwargs={},
        sr_ratio=2,
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        bcg_synthesis_kwargs={},
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        triplane_fg_bg=False,
        encoder_in_channels=3,
        sd_E_ch=64,
        z_channels=3*4,
        sd_E_num_res_blocks=1,
        arch_dit_decoder='DiT2-B/2',
        lrm_decoder=False,
        gs_rendering=False,
        return_all_dit_layers=False,
        *args,
        **kwargs):

    preprocess = None
    clip_dtype = None

    if load_pretrain_encoder:
        if not use_clip:
            if dino_version == 'v1':
                encoder = torch.hub.load(
                    'facebookresearch/dino:main',
                    'dino_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v1 ViT-S{patch_size} encoder ckpt'
                )
            elif dino_version == 'v2':
                encoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v2 {arch_encoder}{patch_size} encoder ckpt'
                )
            elif 'sd' in dino_version:
                if 'mv' in dino_version:
                    if 'lgm' in dino_version:
                        encoder_cls = MVUNet(
                            input_size=256,
                            up_channels=(1024, 1024, 512, 256, 128),
                            up_attention=(True, True, True, False, False),
                            splat_size=128,
                            output_size=512,
                            batch_size=8,
                            num_views=8,
                            gradient_accumulation_steps=1,
                        )
                    elif 'dynaInp' in dino_version:
                        encoder_cls = MVEncoderGSDynamicInp
                    else:
                        encoder_cls = MVEncoder
                    attn_kwargs = {
                        'n_heads': 8,
                        'd_head': 64,
                    }
                else:
                    encoder_cls = Encoder
            else:
                raise NotImplementedError()
        else:
            import clip
            model, preprocess = clip.load("ViT-B/16", device=dist_util.dev())
            model.float()
            clip_dtype = model.dtype
            encoder = getattr(model, 'visual')
            encoder.requires_grad_(False)
            logger.log(f'loaded pre-trained CLIP ViT-B{patch_size} encoder, fixed.')

    elif 'sd' in dino_version:
        attn_kwargs = {}
        if 'mv' in dino_version:
            if 'lgm' in dino_version:
                encoder = LGM_MVEncoder(
                    in_channels=9,
                    up_channels=(1024, 1024, 512, 256, 128),
                    up_attention=(True, True, True, False, False),
                )
            elif 'gs' in dino_version:
                print('using MVEncoderGS')
                encoder_cls = MVEncoderGS
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }
            else:
                print('using MVEncoder')
                encoder_cls = MVEncoder
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }
        else:
            print('using Encoder')
            encoder_cls = Encoder

        Cvae = 8
        if 'lgm' not in dino_version:
            double_z = False
            encoder = encoder_cls(
                double_z=double_z,
                resolution=256,
                in_channels=encoder_in_channels,
                ch=160,
                ch_mult=[1, 1, 2, 2, 4],
                num_res_blocks=2,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=3,
                z_channels=Cvae * 3,
                attn_kwargs=attn_kwargs,
            )

    else:
        encoder = vits.__dict__[arch_encoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            img_size=img_size)

    if triplane_in_chans == -1:
        triplane_in_chans = decoder_in_chans

    triplane_renderer_cls = Triplane

    triplane_decoder = triplane_renderer_cls(
        c_dim,
        image_size,
        img_channels,
        rendering_kwargs=rendering_kwargs,
        out_chans=out_chans,
        triplane_size=triplane_size,
        decoder_in_chans=triplane_in_chans,
        decoder_output_dim=decoder_output_dim,
        sr_kwargs=sr_kwargs,
        bcg_synthesis_kwargs=bcg_synthesis_kwargs,
        lrm_decoder=lrm_decoder)

    if load_pretrain_encoder:
        if dino_version == 'v1':
            vit_decoder = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_{}{}'.format(arch_decoder, patch_size))
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dino:main', 'dino_{}{}".format(
                    arch_decoder, patch_size))
        else:
            vit_decoder = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_{}{}'.format(arch_decoder, patch_size),
                pretrained=decoder_load_pretrained)
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dinov2', 'dinov2_{}{}".format(
                    arch_decoder,
                    patch_size), 'pretrianed=', decoder_load_pretrained)

    elif 'dit' in dino_version:
        from dit.dit_decoder import DiT2_models

        vit_decoder = DiT2_models[arch_dit_decoder](
            input_size=16,
            num_classes=0,
            learn_sigma=False,
            in_channels=embed_dim,
            mixed_prediction=False,
            context_dim=None,
            roll_out=True,
            plane_n=4 if 'gs' in dino_version else 3,
            return_all_layers=return_all_dit_layers,
        )

    else:
        vit_decoder = vits.__dict__[arch_decoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            img_size=img_size)

    decoder_kwargs = dict(
        class_name=ae_classname,
        vit_decoder=vit_decoder,
        triplane_decoder=triplane_decoder,
        cls_token=decoder_cls_token,
        sr_ratio=sr_ratio,
        vae_p=vae_p,
        ldm_z_channels=ldm_z_channels,
        ldm_embed_dim=ldm_embed_dim,
    )

    decoder = dnnlib.util.construct_class_by_name(**decoder_kwargs)

    if use_conf_map:
        confnet = ConfNet(cin=3, cout=1, nf=64, zdim=128)
    else:
        confnet = None

    auto_encoder = AE(
        encoder,
        decoder,
        img_size[0],
        encoder_cls_token,
        decoder_cls_token,
        preprocess,
        use_clip,
        dino_version,
        clip_dtype,
        no_dim_up_mlp=no_dim_up_mlp,
        dim_up_mlp_as_func=dim_up_mlp_as_func,
        uvit_skip_encoder=uvit_skip_encoder,
        confnet=confnet,
    )

    torch.cuda.empty_cache()

    return auto_encoder


def create_3DAE_model_mesh(
        arch_encoder,
        arch_decoder,
        dino_version='v1',
        img_size=[224],
        patch_size=16,
        in_chans=384,
        num_classes=0,
        embed_dim=1024,
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        out_chans=96,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=32,
        encoder_cls_token=False,
        decoder_cls_token=False,
        c_dim=25,
        image_size=128,
        img_channels=3,
        rendering_kwargs={},
        load_pretrain_encoder=False,
        decomposed=True,
        triplane_size=224,
        ae_classname='ViTTriplaneDecomposed',
        use_clip=False,
        sr_kwargs={},
        sr_ratio=2,
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        bcg_synthesis_kwargs={},
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        triplane_fg_bg=False,
        encoder_in_channels=3,
        sd_E_ch=64,
        z_channels=3*4,
        sd_E_num_res_blocks=1,
        arch_dit_decoder='DiT2-B/2',
        lrm_decoder=False,
        gs_rendering=False,
        return_all_dit_layers=False,
        grid_size=128,
        grid_scale=2.005,
        *args,
        **kwargs):

    preprocess = None
    clip_dtype = None

    if load_pretrain_encoder:
        if not use_clip:
            if dino_version == 'v1':
                encoder = torch.hub.load(
                    'facebookresearch/dino:main',
                    'dino_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v1 ViT-S{patch_size} encoder ckpt'
                )
            elif dino_version == 'v2':
                encoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_{}{}'.format(arch_encoder, patch_size))
                logger.log(
                    f'loaded pre-trained dino v2 {arch_encoder}{patch_size} encoder ckpt'
                )
            elif 'sd' in dino_version:
                if 'mv' in dino_version:
                    if 'lgm' in dino_version:
                        encoder_cls = MVUNet(
                            input_size=256,
                            up_channels=(1024, 1024, 512, 256, 128),
                            up_attention=(True, True, True, False, False),
                            splat_size=128,
                            output_size=512,
                            batch_size=8,
                            num_views=8,
                            gradient_accumulation_steps=1,
                        )
                    elif 'dynaInp' in dino_version:
                        encoder_cls = MVEncoderGSDynamicInp
                    else:
                        encoder_cls = MVEncoder
                    attn_kwargs = {
                        'n_heads': 8,
                        'd_head': 64,
                    }
                else:
                    encoder_cls = Encoder
            else:
                raise NotImplementedError()
        else:
            import clip
            model, preprocess = clip.load("ViT-B/16", device=dist_util.dev())
            model.float()
            clip_dtype = model.dtype
            encoder = getattr(model, 'visual')
            encoder.requires_grad_(False)
            logger.log(f'loaded pre-trained CLIP ViT-B{patch_size} encoder, fixed.')

    elif 'sd' in dino_version:
        attn_kwargs = {}
        if 'mv' in dino_version:
            if 'lgm' in dino_version:
                encoder = LGM_MVEncoder(
                    in_channels=9,
                    up_channels=(1024, 1024, 512, 256, 128),
                    up_attention=(True, True, True, False, False),
                )
            elif 'gs' in dino_version:
                print('using MVEncoderGS')
                encoder_cls = MVEncoderGS
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }
            else:
                print('using MVEncoder')
                encoder_cls = MVEncoder
                attn_kwargs = {
                    'n_heads': 8,
                    'd_head': 64,
                }
        else:
            print('using Encoder')
            encoder_cls = Encoder

        Cvae = 8
        if 'lgm' not in dino_version:
            double_z = False
            encoder = encoder_cls(
                double_z=double_z,
                resolution=256,
                in_channels=encoder_in_channels,
                ch=160,
                ch_mult=[1, 1, 2, 2, 4],
                num_res_blocks=2,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=3,
                z_channels=Cvae * 3,
                attn_kwargs=attn_kwargs,
            )

    else:
        encoder = vits.__dict__[arch_encoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            img_size=img_size)

    if triplane_in_chans == -1:
        triplane_in_chans = decoder_in_chans

    triplane_renderer_cls = TriplaneMesh

    triplane_decoder = triplane_renderer_cls(
        c_dim,
        image_size,
        img_channels,
        rendering_kwargs=rendering_kwargs,
        out_chans=out_chans,
        triplane_size=triplane_size,
        decoder_in_chans=triplane_in_chans,
        decoder_output_dim=decoder_output_dim,
        sr_kwargs=sr_kwargs,
        bcg_synthesis_kwargs=bcg_synthesis_kwargs,
        lrm_decoder=True,
        grid_res=grid_size,
        grid_scale=grid_scale,)

    if load_pretrain_encoder:
        if dino_version == 'v1':
            vit_decoder = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_{}{}'.format(arch_decoder, patch_size))
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dino:main', 'dino_{}{}".format(
                    arch_decoder, patch_size))
        else:
            vit_decoder = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_{}{}'.format(arch_decoder, patch_size),
                pretrained=decoder_load_pretrained)
            logger.log(
                'loaded pre-trained decoder',
                "facebookresearch/dinov2', 'dinov2_{}{}".format(
                    arch_decoder,
                    patch_size), 'pretrianed=', decoder_load_pretrained)

    elif 'dit' in dino_version:
        from dit.dit_decoder import DiT2_models

        vit_decoder = DiT2_models[arch_dit_decoder](
            input_size=16,
            num_classes=0,
            learn_sigma=False,
            in_channels=embed_dim,
            mixed_prediction=False,
            context_dim=None,
            roll_out=True,
            plane_n=4 if 'gs' in dino_version else 3,
            return_all_layers=return_all_dit_layers,
        )

    else:
        vit_decoder = vits.__dict__[arch_decoder](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            img_size=img_size)

    decoder_kwargs = dict(
        class_name=ae_classname,
        vit_decoder=vit_decoder,
        triplane_decoder=triplane_decoder,
        cls_token=decoder_cls_token,
        sr_ratio=sr_ratio,
        vae_p=vae_p,
        ldm_z_channels=ldm_z_channels,
        ldm_embed_dim=ldm_embed_dim,
    )

    decoder = dnnlib.util.construct_class_by_name(**decoder_kwargs)

    if use_conf_map:
        confnet = ConfNet(cin=3, cout=1, nf=64, zdim=128)
    else:
        confnet = None

    auto_encoder = AE(
        encoder,
        decoder,
        img_size[0],
        encoder_cls_token,
        decoder_cls_token,
        preprocess,
        use_clip,
        dino_version,
        clip_dtype,
        no_dim_up_mlp=no_dim_up_mlp,
        dim_up_mlp_as_func=dim_up_mlp_as_func,
        uvit_skip_encoder=uvit_skip_encoder,
        confnet=confnet,
    )

    torch.cuda.empty_cache()

    return auto_encoder


def create_Triplane(
        c_dim=25,
        img_resolution=128,
        img_channels=3,
        rendering_kwargs={},
        decoder_output_dim=32,
        *args,
        **kwargs):

    decoder = Triplane(
        c_dim,
        img_resolution,
        img_channels,
        rendering_kwargs=rendering_kwargs,
        create_triplane=True,
        decoder_output_dim=decoder_output_dim)
    return decoder
