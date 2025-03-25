import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, AdaLNCrossSelfAttn_Image, AdaLNCrossSelfAttn, AdaLNCrossSelfAttn_Image_new
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

from ipdb import set_trace as st
from einops import rearrange
from torch.nn import functional as F


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        # st()
        assert embed_dim % num_heads == 0
        # st()
        # self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.Cvae, self.V = vae_local.decoder.superresolution.quantize.Cvae, vae_local.decoder.superresolution.quantize.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        # quant: VectorQuantizer2 = vae_local.quantize
        quant: VectorQuantizer2 = vae_local.decoder.superresolution.quantize

        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        # self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        # nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        # change the input pooler output to the same size as the class embedding (768 -> 1024)
        # self.pooler_emb = nn.Linear(768, self.C)
        self.pooler_emb = nn.Linear(1024, self.C)
        nn.init.trunc_normal_(self.pooler_emb.weight.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pooler_emb.bias.data, mean=0, std=init_std)
        # st()
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        # pos_1LC = []
        # for i, pn in enumerate(self.patch_nums):
        #     pe = torch.empty(1, pn*pn, self.C)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        # pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        # assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn*3, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L*3, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        # plane embedding
        self.plane_embed = nn.Embedding(3, self.C)
        nn.init.trunc_normal_(self.plane_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        # shared_aln is False here
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            # AdaLNSelfAttn(
            #     cond_dim=self.D, shared_aln=shared_aln,
            #     block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
            #     attn_l2_norm=attn_l2_norm,
            #     flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            # )
            # AdaLNCrossSelfAttn_Image(
            #     cond_dim=self.D, shared_aln=shared_aln,
            #     block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
            #     attn_l2_norm=attn_l2_norm,
            #     flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            # )
            AdaLNCrossSelfAttn_Image_new(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        # d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        d: torch.Tensor = torch.cat([torch.full((pn*pn*3,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L * 3, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        # st()
        self.register_buffer('lvl_1L', lvl_1L)
        # attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L * 3, self.L * 3)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        # create a buffer for cross attention
        # attn_bias_cross_for_masking = torch.zeros((1, 1, self.L * 3, 77))
        # self.register_buffer('attn_bias_cross_for_masking', attn_bias_cross_for_masking.contiguous())
        # add attention mask for cross attention

        # st()
        p = []
        for pn in self.patch_nums:
            p.append(torch.tensor([0]).repeat(pn*pn))
            p.append(torch.tensor([1]).repeat(pn*pn))
            p.append(torch.tensor([2]).repeat(pn*pn))
        p_exp = torch.cat(p).unsqueeze(0)
        plane_1L = p_exp.contiguous()
        self.register_buffer('plane_1L', plane_1L)
        # st()

        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        sos = cond_BD = self.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            # FIXME: donot use conditional sampling here
            # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            return latent_after_vit
        
        # return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    @torch.no_grad()
    def autoregressive_infer_3D_VAR_image(
        self, B: int, clip_image_embeddings: Optional[Union[int, torch.LongTensor]],
        dino_image_embeddings: Optional[Union[int, torch.LongTensor]],
        pooler_output: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     # label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=self.rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        # st()
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkk",pooler_output.shape)
        sos = cond_BD = self.pooler_emb(pooler_output)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        

        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()

            # hack: get the dtype if mixed precision is used
            # temp = x.new_ones(8, 8)
            # main_type = torch.matmul(temp, temp).dtype
            
            # # QUESTION: this part change the data type to torch.float16, why?
            # x = x.to(dtype=torch.float16)
            # cond_BD_or_gss = cond_BD_or_gss.to(dtype=torch.float16)
            # text_embeddings = text_embeddings.to(dtype=torch.float16)


            AdaLNCrossSelfAttn_Image.forward
            for b in self.blocks:
                # x = b(x=x, cond_BD=cond_BD_or_gss, condition=text_embeddings, attn_bias=None)
                x = b(x=x, cond_BD=cond_BD_or_gss, clip_condition=clip_image_embeddings, dino_condition=dino_image_embeddings, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            # FIXME: donot use conditional sampling here
            # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            # st()
            return latent_after_vit

    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_image(
        self, B: int, clip_image_embeddings: Optional[Union[int, torch.LongTensor]],
        dino_image_embeddings: Optional[Union[int, torch.LongTensor]],
        pooler_output: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     # label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=self.rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        # st()
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkk",pooler_output.shape)

        # classifier free guidance
        import numpy as np
        # empty_pooler_output = torch.zeros_like(pooler_output).to(pooler_output.device)[0].unsqueeze(0)
        # empty_clip_image_embeddings = torch.zeros_like(clip_image_embeddings).to(clip_image_embeddings.device)[0].unsqueeze(0)
        # empty_dino_image_embeddings = torch.zeros_like(dino_image_embeddings).to(dino_image_embeddings.device)[0].unsqueeze(0)
        # empty_clip_image_embeddings = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_embedding.npy")).to(pooler_output.device).unsqueeze(0)
        # empty_pooler_output = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        empty_pooler_output = torch.from_numpy(np.load("./empty_dino_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        empty_dino_image_embeddings = torch.from_numpy(np.load("./empty_dino_embedding.npy"))[1:, :].to(pooler_output.device).unsqueeze(0)
        # pooler_output = torch.cat((empty_pooler_output.expand(pooler_output.shape), pooler_output), dim=0)
        # clip_image_embeddings = torch.cat((empty_clip_image_embeddings.expand(clip_image_embeddings.shape), clip_image_embeddings), dim=0)
        # dino_image_embeddings = torch.cat((empty_dino_image_embeddings.expand(dino_image_embeddings.shape), dino_image_embeddings), dim=0)
        pooler_output = torch.cat((pooler_output, empty_pooler_output.expand(pooler_output.shape)), dim=0)
        # clip_image_embeddings = torch.cat((clip_image_embeddings, empty_clip_image_embeddings.expand(clip_image_embeddings.shape)), dim=0)
        dino_image_embeddings = torch.cat((dino_image_embeddings, empty_dino_image_embeddings.expand(dino_image_embeddings.shape)), dim=0)


        sos = cond_BD = self.pooler_emb(pooler_output)
        # sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        # classifier free guidance
        sos = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        # next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # classifier free guidance
        next_token_map = sos.expand(2 * B, 3, -1) + lvl_pos[:, :3]
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        

        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()

            # hack: get the dtype if mixed precision is used
            # temp = x.new_ones(8, 8)
            # main_type = torch.matmul(temp, temp).dtype
            
            # # QUESTION: this part change the data type to torch.float16, why?
            # x = x.to(dtype=torch.float16)
            # cond_BD_or_gss = cond_BD_or_gss.to(dtype=torch.float16)
            # text_embeddings = text_embeddings.to(dtype=torch.float16)


            # AdaLNCrossSelfAttn_Image.forward
            AdaLNCrossSelfAttn_Image_new.forward
            for b in self.blocks:
                # x = b(x=x, cond_BD=cond_BD_or_gss, condition=text_embeddings, attn_bias=None)
                # x = b(x=x, cond_BD=cond_BD_or_gss, clip_condition=clip_image_embeddings, dino_condition=dino_image_embeddings, attn_bias=None)
                x = b(x=x, cond_BD=cond_BD_or_gss, dino_condition=dino_image_embeddings, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # classfier free guidance
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            # st()
            return latent_after_vit
        
    @torch.no_grad()
    # def autoregressive_infer_cfg_3D_VAR_image_l2norm(
    #     self, B: int, clip_image_embeddings: Optional[Union[int, torch.LongTensor]],
    #     dino_image_embeddings: Optional[Union[int, torch.LongTensor]],
    #     pooler_output: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, g_seed: Optional[int] = None,
    #     more_smooth=False,
    # ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
    def autoregressive_infer_cfg_3D_VAR_image_l2norm(
        self, B: int,
        dino_image_embeddings: Optional[Union[int, torch.LongTensor]],
        pooler_output: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     # label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=self.rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        # st()
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkk",pooler_output.shape)

        # classifier free guidance
        import numpy as np
        # empty_pooler_output = torch.zeros_like(pooler_output).to(pooler_output.device)[0].unsqueeze(0)
        # empty_clip_image_embeddings = torch.zeros_like(clip_image_embeddings).to(clip_image_embeddings.device)[0].unsqueeze(0)
        # empty_dino_image_embeddings = torch.zeros_like(dino_image_embeddings).to(dino_image_embeddings.device)[0].unsqueeze(0)
        # empty_clip_image_embeddings = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_embedding.npy")).to(pooler_output.device).unsqueeze(0)
        # empty_pooler_output = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        # empty_dino_image_embeddings = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_dino_embedding.npy"))[1:, :].to(pooler_output.device).unsqueeze(0)
        # empty_clip_image_embeddings = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_embedding.npy")).to(pooler_output.device).unsqueeze(0)
        empty_pooler_output = torch.from_numpy(np.load("./empty_dino_pooler_output.npy")).to(pooler_output.device).unsqueeze(0)
        empty_dino_image_embeddings = torch.from_numpy(np.load("./empty_dino_embedding.npy"))[1:, :].to(pooler_output.device).unsqueeze(0)
        # pooler_output = torch.cat((empty_pooler_output.expand(pooler_output.shape), pooler_output), dim=0)
        # clip_image_embeddings = torch.cat((empty_clip_image_embeddings.expand(clip_image_embeddings.shape), clip_image_embeddings), dim=0)
        # dino_image_embeddings = torch.cat((empty_dino_image_embeddings.expand(dino_image_embeddings.shape), dino_image_embeddings), dim=0)
        pooler_output = torch.cat((pooler_output, empty_pooler_output.expand(pooler_output.shape)), dim=0)
        # clip_image_embeddings = torch.cat((clip_image_embeddings, empty_clip_image_embeddings.expand(clip_image_embeddings.shape)), dim=0)
        dino_image_embeddings = torch.cat((dino_image_embeddings, empty_dino_image_embeddings.expand(dino_image_embeddings.shape)), dim=0)


        sos = cond_BD = self.pooler_emb(pooler_output)
        # sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        # classifier free guidance
        sos = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        # next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # classifier free guidance
        next_token_map = sos.expand(2 * B, 3, -1) + lvl_pos[:, :3]
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        
        

        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()

            # hack: get the dtype if mixed precision is used
            # temp = x.new_ones(8, 8)
            # main_type = torch.matmul(temp, temp).dtype
            
            # # QUESTION: this part change the data type to torch.float16, why?
            # x = x.to(dtype=torch.float16)
            # cond_BD_or_gss = cond_BD_or_gss.to(dtype=torch.float16)
            # text_embeddings = text_embeddings.to(dtype=torch.float16)


            # AdaLNCrossSelfAttn_Image.forward
            AdaLNCrossSelfAttn_Image_new.forward
            for b in self.blocks:
                # x = b(x=x, cond_BD=cond_BD_or_gss, condition=text_embeddings, attn_bias=None)
                # x = b(x=x, cond_BD=cond_BD_or_gss, clip_condition=clip_image_embeddings, dino_condition=dino_image_embeddings, attn_bias=None)
                x = b(x=x, cond_BD=cond_BD_or_gss, dino_condition=dino_image_embeddings, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if si == 0:
                g_BL = idx_Bl
            else:
                g_BL = torch.cat((g_BL, idx_Bl), dim=1)
            # st()
            # print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            # assert not more_smooth, "more_smooth is not supported in l2norm"
            if not more_smooth: # this is the default case
                # h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                # refer to llamagen
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                # h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ embedding.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # classfier free guidance
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            # st()
            # return latent_after_vit
            return  latent_after_vit, g_BL

    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_text(
        self, B: int, text_embeddings: Optional[Union[int, torch.LongTensor]],
        pooler_output: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, g_seed: Optional[int] = None,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     # label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=self.rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        # st()
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        sos = cond_BD = self.pooler_emb(pooler_output)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        

        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()

            # hack: get the dtype if mixed precision is used
            # temp = x.new_ones(8, 8)
            # main_type = torch.matmul(temp, temp).dtype
            
            # # QUESTION: this part change the data type to torch.float16, why?
            # x = x.to(dtype=torch.float16)
            # cond_BD_or_gss = cond_BD_or_gss.to(dtype=torch.float16)
            # text_embeddings = text_embeddings.to(dtype=torch.float16)


            AdaLNCrossSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, condition=text_embeddings, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            # FIXME: donot use conditional sampling here
            # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            # st()
            return latent_after_vit
        
    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_condition(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]], condition,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        sos = cond_BD = self.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, condition=condition, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            # FIXME: donot use conditional sampling here
            # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            print("sjlksldhalfhkfhkl", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                #     st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            return latent_after_vit
        
        # return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    

    @torch.no_grad()
    def reconstruct_gt_Bl_idx_multiscale(self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, gt_idx: Optional[torch.LongTensor] = None, scale: int = 10
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        cur_L = 0

        f_hat_list = [torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device), \
                      torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device), \
                      torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device)]
        
        for b in self.blocks: b.attn.kv_caching(True)
        # st()
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn*3

            idx_Bl = gt_idx[:, cur_L-pn*pn*3:cur_L]
            # True here
            if not more_smooth: # this is the default case
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                pass
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                # st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], _ = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                # st()
                # next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            if scale > len(self.patch_nums):
                raise ValueError(f"scale should be less than {len(self.patch_nums)}")
            if si==scale - 1:
                break

        for b in self.blocks: b.attn.kv_caching(False)
        # st()
        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)

            return latent_after_vit
        

    @torch.no_grad()
    def reconstruct_gt_Bl_idx(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, gt_idx: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        # sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        # sos = sos.repeat(1, 3, 1)
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        # next_token_map = sos.expand(B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        # st()
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device), \
                      torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device), \
                      torch.zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1], device=gt_idx.device)]
        
        for b in self.blocks: b.attn.kv_caching(True)
        # aaa = (1, 2, 3, 4, 5, 6, 8, 10, 13)
        # for si, pn in enumerate(aaa):   # si: i-th segment
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            # cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            # x = next_token_map
            # print("x",x.shape)
            # st()
            # AdaLNSelfAttn.forward
            # for b in self.blocks:
            #     x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            # logits_BlV = self.get_logits(x, cond_BD)
            # st()
            # t = cfg * ratio
            # FIXME: donot use conditional sampling here
            # logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            
            # idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # st()
            idx_Bl = gt_idx[:, cur_L-pn*pn*3:cur_L]
            # st()
            # print("lkjlsjlkjflksjk", idx_Bl.shape)
            # st()
            # True here
            if not more_smooth: # this is the default case
                # refer to llamagen
                # print("si:", si)
                embedding = F.normalize(self.vae_quant_proxy[0].embedding.weight, p=2, dim=-1)
                h_BChw_concate = embedding[idx_Bl]   # B, l, Cvae
                # st()
                # h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                pass
                # h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                # if si == 5:
                # st()
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], _ = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                # next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            # st()
            # next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            # if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                # next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        # st()
        for b in self.blocks: b.attn.kv_caching(False)
        # st()
        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            return latent_after_vit
        
        # return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    @torch.no_grad()
    def autoregressive_infer_cfg_3D_VAR_w_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        # sos = cond_BD = self.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1)
        sos = sos.repeat(1, 3, 1)
        # st()
        
        # lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC 
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC + self.plane_embed(self.plane_1L)
        # FIXME: find out why use 2 * B here
        # next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        # st()
        next_token_map = sos.expand(2 * B, 3, -1) + lvl_pos[:, :3] 
        # st()
        cur_L = 0
        
        # f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_list = [sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]), \
                      sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])]
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            # cur_L += pn*pn
            cur_L += pn*pn*3
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # print("x",x.shape)
            # st()
            AdaLNSelfAttn.forward
            # print("sisisisisissisisi",si)
            # print("x.shape", x.shape)
            # print("cond_BD_or_gss.shape", cond_BD_or_gss.shape)
            # if si == 8:
            #     st()
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            # st()
            t = cfg * ratio
            # FIXME: donot use conditional sampling here
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # st()
            # True here
            if not more_smooth: # this is the default case
                h_BChw_concate = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw_concate = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            # st()
            h_BChw_list = h_BChw_concate.split(int(h_BChw_concate.shape[1]/3), dim=1)
            # h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            next_token_map_list = []
            for i in range(3):
                h_BChw = h_BChw_list[i].transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat_list[i], next_token_map_single_plane = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_list[i], h_BChw)
                next_token_map_list.append(next_token_map_single_plane.view(B, self.Cvae, -1).transpose(1, 2))
            # if si==3:
            #     st()
            next_token_map = torch.cat(next_token_map_list, dim=1)
            # print("next_token_map",next_token_map.shape)

            if si != self.num_stages_minus_1:   # prepare for next stage
                # next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                # next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + 3 * self.patch_nums[si+1] ** 2]
                # FIXME: donot use CFG here
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        # st()
        for b in self.blocks: b.attn.kv_caching(False)

        # TODO: add the 3 channel f_hat to the decoder and output the 3D triplane
        # To 3*B, Cvae, H, W
        f_hat_all = torch.zeros((3 * f_hat_list[0].shape[0], f_hat_list[0].shape[1], f_hat_list[0].shape[2], f_hat_list[0].shape[3]), device=f_hat_list[0].device, dtype=f_hat_list[0].dtype)
        f_hat_all[0::3] = f_hat_list[0]
        f_hat_all[1::3] = f_hat_list[1]
        f_hat_all[2::3] = f_hat_list[2]

        with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, cache_enabled=True
            ):  # only handles the execusion, not data type
            # st()
            f_hat_all = self.vae_proxy[0].decoder.superresolution['post_quant_conv'](f_hat_all.to(torch.bfloat16))

            # only for pretrained model
            # f_hat = self.superresolution['post_quant_conv_for_pretrained'](f_hat)
            # st()
            # TODO_done (chen): 1. add code for upsample 2.add code for foward_vit_decoder 3. return forwar_vit_decoder latent
            f_hat_all = f_hat_all.reshape(f_hat_all.shape[0] // 3, -1, f_hat_all.shape[-2], f_hat_all.shape[-1]) # (B * 3, 32, 16, 16) -> (B, 32 * 3, 16, 16)
            # TODO_done (chen) see the structure of the ldm_upsample
            # PatchEmbedTriplane.forward
            # st()
            
            f_hat_all = self.vae_proxy[0].decoder.superresolution['ldm_upsample'](f_hat_all)
            f_hat_all = self.vae_proxy[0].decoder.forward_vit_decoder(f_hat_all, 224)
            latent_after_vit = self.vit_decode_postprocess(f_hat_all)
            # st()
            # here we get decode without triplane
            return latent_after_vit
        
    def vit_decode_postprocess(self, latent_from_vit):
        # st()
        # False here
        if self.vae_proxy[0].decoder.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # st()
        # latent = unflatten_token(
        #     latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        # 每个plane在这个阶段独立decode，互不影响
        # Decoder.forward
        # st()
        # TODO: 1. unflatten token here 
        # st()
        latent = self.unflatten_token(latent_from_vit)  # B 3 h w vit_decoder.embed_dim
        # st()
        # latent = latent_from_vit
        latent = self.vae_proxy[0].decoder.superresolution['conv_sr'](latent)  # still B 3C H W
        # st()
        # True here
        if not self.vae_proxy[0].decoder.D_roll_out_input:
            # st()
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        return latent
    
    def unflatten_token(self, x, p=None):
        B, L, C = x.shape
        x = x.reshape(B, 3, L // 3, C)

        # False here
        if self.vae_proxy[0].decoder.cls_token:  # TODO, how to better use cls token
            x = x[:, :, 1:]  # B 3 256 C
        # st()
        h = w = int((x.shape[2])**.5)
        assert h * w == x.shape[2]

        # True here
        if p is None:
            x = x.reshape(shape=(B, 3, h, w, -1))
            # True here
            if not self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(
                    x, 'b n h w c->(b n) c h w'
                )  # merge plane into Batch and prepare for rendering
            else:
                x = rearrange(
                    x, 'b n h w c->b c h (n w)'
                )  # merge plane into Batch and prepare for rendering
        else:
            x = x.reshape(shape=(B, 3, h, w, p, p, -1))
            if self.vae_proxy[0].decoder.D_roll_out_input:
                x = rearrange(
                    x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                )  # merge plane into Batch and prepare for rendering
            else:
                x = rearrange(
                    x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                )  # merge plane into Batch and prepare for rendering

        return x

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    # For concatenta version input
    # def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
    # def forward(self, pooler_output: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
    # def forward(self, pooler_output: torch.LongTensor, condition: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
    # def forward(self, pooler_output: torch.LongTensor, clip_condition: torch.LongTensor, dino_condition: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
    # classifier free guidance
    # def forward(self, pooler_output, clip_condition, dino_condition, x_BLCv_wo_first_l, empty_pooler_output, empty_clip_image_embedding, empty_dino_embedding) -> torch.Tensor:    
    def forward(self, pooler_output, dino_condition, x_BLCv_wo_first_l, empty_pooler_output, empty_dino_embedding) -> torch.Tensor:  
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        # st()
        # TODO: add positional embedding for each plane
        # get x_BLC shape: B, 679*3, 1024
        # st()
        # bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L * 3)
        # st()
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            # st()
            # label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            # sos = cond_BD = self.class_emb(label_B)
            # classifier free guidance
            # random drop the input text
            replace_pooler = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            pooler_output[replace_pooler] = empty_pooler_output
            # replace_condition = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            # clip_condition[replace_condition] = empty_clip_image_embedding
            replace_condition = torch.rand(B, device=pooler_output.device) < self.cond_drop_rate
            dino_condition[replace_condition] = empty_dino_embedding

            # generate sos according to the given pooler output
            sos = cond_BD = self.pooler_emb(pooler_output)
            # st()
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            sos = sos.repeat(1, 3, 1)
            # False here
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            
            # self.lvl_embed's shape is [10, 1024], it is a positional embedding for each scale
            # self.pos_1LC's shape is [680, 1024], it is a positional embedding for each seperate token
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC

            # add positional embedding for each plane
            x_BLC = x_BLC + self.plane_embed(self.plane_1L[:, :ed].expand(B, -1))
        # st()
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        # attn_bias_cross = self.attn_bias_cross_for_masking
        # here self.shared_ada_lin is an Identity layer
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        # QUESTION: this part change the data type to torch.float16, why?
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        # attn_bias_cross = attn_bias_cross.to(dtype=main_type)
        
        # AdaLNSelfAttn.forward
        # AdaLNCrossSelfAttn_Image.forward
        AdaLNCrossSelfAttn_Image_new.forward
        # st()
        for i, b in enumerate(self.blocks):
            # x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            # add condition
            # st()
            # x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, condition=condition, attn_bias=attn_bias, attn_bias_cross=attn_bias_cross)
            # x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, clip_condition=clip_condition, dino_condition=dino_condition, attn_bias=attn_bias)
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, dino_condition=dino_condition, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        # st()
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    

    # def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
    #     """
    #     :param label_B: label_B
    #     :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
    #     :return: logits BLV, V is vocab_size
    #     """
    #     # st()

    #     # st()
    #     bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
    #     # st()
    #     B = x_BLCv_wo_first_l.shape[0]
    #     with torch.cuda.amp.autocast(enabled=False):
    #         label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
    #         sos = cond_BD = self.class_emb(label_B)
    #         sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
    #         if self.prog_si == 0: x_BLC = sos
    #         else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
    #         # self.lvl_embed's shape is [10, 1024], it is a positional embedding for each scale
    #         # self.pos_1LC's shape is [680, 1024], it is a positional embedding for each seperate token
    #         x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC

    #         # add positional embedding for each plane
    #         # x_BLC = x_BLC + self.plane_embed(self.plane_1L[:, :ed].repeat(int(B/3), 1))
        
    #     attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
    #     # here self.shared_ada_lin is an Identity layer
    #     cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
    #     # hack: get the dtype if mixed precision is used
    #     temp = x_BLC.new_ones(8, 8)
    #     main_type = torch.matmul(temp, temp).dtype
        
    #     # QUESTION: this part change the data type to torch.float16, why?
    #     x_BLC = x_BLC.to(dtype=main_type)
    #     cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
    #     attn_bias = attn_bias.to(dtype=main_type)
        
    #     AdaLNSelfAttn.forward
    #     # st()
    #     for i, b in enumerate(self.blocks):
    #         x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
    #     x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
    #     if self.prog_si == 0:
    #         if isinstance(self.word_embed, nn.Linear):
    #             x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
    #         else:
    #             s = 0
    #             for p in self.word_embed.parameters():
    #                 if p.requires_grad:
    #                     s += p.view(-1)[0] * 0
    #             x_BLC[0, 0, 0] += s
    #     return x_BLC    # logits BLV, V is vocab_size
    
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
