from typing import Tuple
import torch.nn as nn

from vit.quant import VectorQuantizer2
from .var import VAR
from .vqvae import VQVAE

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from nsr.script_util import create_3DAE_model, create_3DAE_model_mesh

from ipdb import set_trace as st


def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    # TODO: add my VQVAE model
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp


def build_vae_var_3D_VAR(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1, args=None    # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    # heads = depth
    heads = 16
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    # TODO: add my VQVAE model
    # st()
    args.LN3DiffConfig.img_size = [args.LN3DiffConfig.image_size_encoder]
    # st()
    if args.flexicubes == False:
        vae_local = create_3DAE_model(**args_to_dict(args.LN3DiffConfig,
                       encoder_and_nsr_defaults().keys())).to(device)
    else:
        # st()
        vae_local = create_3DAE_model_mesh(**args_to_dict(args.LN3DiffConfig,
                       encoder_and_nsr_defaults().keys())).to(device)
        vae_local.decoder.triplane_decoder.init_flexicubes_geometry(device=device, fovy=43)
    # st()
    var_wo_ddp = VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp

def args_to_dict(args, keys):
    for k in keys:
        if not hasattr(args, k):
            print(f"Missing key {k} in args")
    return {k: getattr(args, k) for k in keys}

def encoder_and_nsr_defaults():
    """
    Defaults for image training.
    """
    # ViT configs
    res = dict(
        dino_version='v1',
        encoder_in_channels=3,
        img_size=[224],
        patch_size=16,  # ViT-S/16
        in_chans=384,
        num_classes=0,
        embed_dim=384,  # Check ViT encoder dim
        depth=6,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        # img_resolution=128,  # Output resolution.
        cls_token=False,
        # image_size=128,  # rendered output resolution.
        # img_channels=3,  # Number of output color channels.
        encoder_cls_token=False,
        decoder_cls_token=False,
        sr_kwargs={},
        sr_ratio=2,
        # sd configs
    )
    # Triplane configs
    res.update(model_encoder_defaults())
    res.update(nsr_decoder_defaults())
    res.update(
        ae_classname='vit.vit_triplane.ViTTriplaneDecomposed')  # if add SR
    return res

def model_encoder_defaults():

    return dict(
        use_clip=False,
        arch_encoder="vits",
        arch_decoder="vits",
        load_pretrain_encoder=False,
        encoder_lr=1e-5,
        encoder_weight_decay=
        0.001,  # https://github.com/google-research/vision_transformer
        no_dim_up_mlp=False,
        dim_up_mlp_as_func=False,
        decoder_load_pretrained=True,
        uvit_skip_encoder=False,
        # vae ldm
        vae_p=1,
        ldm_z_channels=4,
        ldm_embed_dim=4,
        use_conf_map=False,
        # sd E, lite version by default
        sd_E_ch=64,
        z_channels=3*4,
        sd_E_num_res_blocks=1,
        # vit_decoder
        arch_dit_decoder='DiT2-B/2',
        return_all_dit_layers=False,
        # sd D
        # sd_D_ch=32,
        # sd_D_res_blocks=1,
        # sd_D_res_blocks=1,
        lrm_decoder=False,
        gs_rendering=False,
    )

def nsr_decoder_defaults():
    res = {
        'decomposed': False,
    }  # TODO, add defaults for all nsr
    res.update(triplane_decoder_defaults())  # triplane by default now
    res.update(vit_decoder_defaults())  # type: ignore
    return res

def triplane_decoder_defaults():
    opts = dict(
        triplane_fg_bg=False,
        cfg='shapenet',
        density_reg=0.25,
        density_reg_p_dist=0.004,
        reg_type='l1',
        triplane_decoder_lr=0.0025,  # follow eg3d G lr
        super_resolution_lr=0.0025,
        # triplane_decoder_wd=0.1,
        c_scale=1,
        nsr_lr=0.02,
        triplane_size=224,
        decoder_in_chans=32,
        triplane_in_chans=-1,
        decoder_output_dim=3,
        out_chans=96,
        c_dim=25,  # Conditioning label (C) dimensionality.
        # ray_start=0.2,
        # ray_end=2.2,
        ray_start=0.6,  # shapenet default
        ray_end=1.8,
        rendering_kwargs={},
        sr_training=False,
        bcg_synthesis=False,  # from panohead
        bcg_synthesis_kwargs={},  # G_kwargs.copy()
        #
        image_size=128,  # raw 3D rendering output resolution.
        patch_rendering_resolution=45,
    )

    # else:
    #     assert False, "Need to specify config"

    # opts = dict(opts)
    # opts.pop('cfg')

    return opts

def vit_decoder_defaults():
    res = dict(
        vit_decoder_lr=1e-5,  # follow eg3d G lr
        vit_decoder_wd=0.001,
    )
    return res