import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
# from models import VAR, VQVAE, VectorQuantizer2
from models import VAR
from vit.quant import VectorQuantizer2
from nsr.script_util import AE as VQVAE
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

from ipdb import set_trace as st

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        dino_image_processor = None, dino_image_model = None
    ):
        super(VARTrainer, self).__init__()
        
        # self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.decoder.superresolution.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        # st()
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        # self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        self.loss_weight = torch.ones(1, self.L * 3, device=device) / (self.L * 3)
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        # for i, pn in enumerate(patch_nums):
        #     self.begin_ends.append((cur, cur + pn * pn))
        #     cur += pn*pn
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn * 3))
            cur += pn*pn * 3
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        self.dino_image_processor = dino_image_processor
        self.dino_image_model = dino_image_model
        # if dino_image_processor is not None:
        #     self.dino_image_processor = dino_image_processor
        # if dino_image_model is not None:
        #     self.dino_image_model = dino_image_model
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval() 
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    @torch.no_grad()
    def eval_ep_3D_VAR(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        # for inp_B3HW, label_B in ld_val:
        cfg = 4
        more_smooth = False
        for data in ld_val:
            # B, V = label_B.shape[0], self.vae_local.vocab_size
            # st()
            # B, V = data["image_clip_embedding"].shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
            # clip_image_embeddings = data["image_clip_embedding"].to(dist.get_device(), non_blocking=True)
            # pooler_output = data["image_clip_pooler_output"].to(dist.get_device(), non_blocking=True)
            # dino_embeddings = data["image_dino_embedding"].to(dist.get_device(), non_blocking=True)
            B, V = data["image_dino_embedding"].shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
            # B, V = data["single_image"].shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
            # clip_image_embeddings = data["image_clip_embedding"].to(dist.get_device(), non_blocking=True)
            # single_image = data['single_image']

            
            # with torch.no_grad():
            #     dino_image = self.dino_image_processor(images=single_image, return_tensors="pt")
            #     dinomodel_output = self.dino_image_model(**dino_image)
            #     dino_embeddings = dinomodel_output.last_hidden_state[:,1:,:].to(dist.get_device(), non_blocking=True)
            #     pooler_output = dinomodel_output.pooler_output.to(dist.get_device(), non_blocking=True)

            pooler_output = data["image_dino_pooler_output"].to(dist.get_device(), non_blocking=True)
            dino_embeddings = data["image_dino_embedding"].to(dist.get_device(), non_blocking=True)
            caption = data["caption"]
            # st()
            with torch.inference_mode():
                # st()
                # triplane = self.var_wo_ddp.autoregressive_infer_3D_VAR_image(B=B, clip_image_embeddings=clip_image_embeddings, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, cfg=cfg, top_k=900, top_p=0.95, more_smooth=more_smooth)
                # calssifier free guidance
                # triplane = self.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_image(B=B, clip_image_embeddings=clip_image_embeddings, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, cfg=cfg, top_k=900, top_p=0.95, more_smooth=more_smooth)
                # triplane = self.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_image_l2norm(B=B, clip_image_embeddings=clip_image_embeddings, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, cfg=cfg, top_k=900, top_p=0.95, more_smooth=more_smooth)
                triplane = self.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_image_l2norm(B=B, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, cfg=cfg, top_k=900, top_p=0.95, more_smooth=more_smooth)
            break
            # inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            # label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            # gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            # gt_BL = torch.cat(gt_idx_Bl, dim=1)
            # x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            # self.var_wo_ddp.forward
            # logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            # L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            # L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            # acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            # acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            # tot += B
        self.var_wo_ddp.train(training)

        return triplane, caption, data
        
        # stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        # dist.allreduce(stats)
        # tot = round(stats[-1].item())
        # stats /= tot
        # L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        
        # return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    # def train_step(
    #     self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
    #     inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    # ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
    # classifier free guidance
    # def train_step(
    #     self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
    #     inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    #     empty_pooler_output, empty_clip_image_embedding, empty_dino_image_embedding
    # ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
    # def train_step(
    #     self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
    #     inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    #     empty_pooler_output, empty_dino_image_embedding
    # ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
        empty_pooler_output, empty_dino_image_embedding
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        # st()
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        # from guided_diffusion import dist_util
        # print("local_rank", dist_util.dev())
        # forward
        # st()
        # TODO (done): check how to obtain label_B here, what's the function of class label
        # label_B is the ground truth label that loaded from the ImageNet dataset
        # B, V = label_B.shape[0], self.vae_local.vocab_size
        # each concatenated image contains 12 images, which will be devided to two groups, each group contains 6 images
        # for each group, the image will be encoded to 3 planes, so bs=1 --> 12 images/labels --> 6 planes
        B, V = label_B.shape[0], self.vae_local.decoder.superresolution.quantize.vocab_size
        self.var.require_backward_grad_sync = stepping
        # st()
        
        # get indices
        # TODO (done): how to get indices from input image
        # This part need not to modified
        # st()
        # torch.autograd.set_detect_anomaly(True)
        # st()
        # gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        
        # # TODO: remove for loop to speed up the training 
        # # concate three plane for each sample
        # # gt_idx_Bl_saperate: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        # group_size = 3
        # gt_Bl_concate_list = []
        # for i in range(len(gt_idx_Bl)):
        #     N = gt_idx_Bl[i].shape[-1]
        #     gt_Bl_concate_list.append(gt_idx_Bl[i].reshape(-1, group_size * N))
        # gt_BL = torch.cat(gt_Bl_concate_list, dim=1)

        # # test code: save the gt_idx_Bl
        # # torch.save(gt_BL, './gt_idx_Bl.pth')
        # # st()

        # # gt_BL = torch.cat(gt_idx_Bl, dim=1)
        # # get input for VAR (BS, 679, 32)
        # # TODO (done): how to get var input
        # # ensure that the iput to VAR is the prediction of VQVAE, rather than the VAR output
        # # this implementation is helpful to train the network, but in inference time, we don't have VQVAE output, so \
        # # we use VAR output as the input to VAR
        # # st()
        # # x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        # x_BLCv_wo_first_l_separate: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        # group_size = 3
        # x_BLCv_wo_first_l = []
        # for i in range(len(x_BLCv_wo_first_l_separate)):
        #     N = x_BLCv_wo_first_l_separate[i].shape[1]
        #     x_BLCv_wo_first_l.append(x_BLCv_wo_first_l_separate[i].reshape(-1, group_size * N, x_BLCv_wo_first_l_separate[i].shape[-1]))
        # x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=1) # B, 679*3, 32
        # st()
        
        gt_BL = inp_B3HW["gt_BL"].to(dist.get_device(), non_blocking=True)
        x_BLCv_wo_first_l = inp_B3HW["x_BLCv_wo_first_l"].to(dist.get_device(), non_blocking=True)

        # =====================generate latent online===================== #
        # inp = inp_B3HW['img_to_encoder']
        # inp = inp.to(dist.get_device(), non_blocking=True)
        # # st()
        # # inp = inp.to(args.device, non_blocking=True)
        
        # # only encode the half data
        # indices = torch.cat([torch.arange(i, i + 6) for i in range(0, inp.size(0), 12)])
        # inp = inp[indices]

        # with torch.no_grad():
        #     gt_idx_Bl = self.vae_local.img_to_idxBl(inp)

        # group_size = 3
        # gt_Bl_concate_list = []
        # for i in range(len(gt_idx_Bl)):
        #     N = gt_idx_Bl[i].shape[-1]
        #     gt_Bl_concate_list.append(gt_idx_Bl[i].reshape(-1, group_size * N))
        # # TODO: save gt_BL and x_BLCv_wo_first_l
        # gt_BL = torch.cat(gt_Bl_concate_list, dim=1)

        # with torch.no_grad():
        #     x_BLCv_wo_first_l_separate = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        # group_size = 3
        # x_BLCv_wo_first_l = []
        # for i in range(len(x_BLCv_wo_first_l_separate)):
        #     N = x_BLCv_wo_first_l_separate[i].shape[1]
        #     x_BLCv_wo_first_l.append(x_BLCv_wo_first_l_separate[i].reshape(-1, group_size * N, x_BLCv_wo_first_l_separate[i].shape[-1]))
        # x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=1) # B, 679*3, 32
        # =====================generate latent online===================== #

        # =====================generate condition online================================ #
        # single_image = inp_B3HW['single_image']
        # with torch.no_grad():
        #     dino_image = self.dino_image_processor(images=single_image, return_tensors="pt")
        #     dinomodel_output = self.dino_image_model(**dino_image)
        #     dino_image_embedding = dinomodel_output.last_hidden_state[:,1:,:]
        #     dino_image_pooler_output = dinomodel_output.pooler_output
        # =====================generate condition online================================ #
        # st()





        # load text encoder to encode the text
        # from transformers import CLIPTextModel, CLIPTokenizer
        # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # # To the GPU
        # # st()
        # text_encoder = text_encoder.to(dist.get_device())
        # text_input = tokenizer(inp_B3HW["caption"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # text_embeddings = text_encoder(text_input.input_ids.to(dist.get_device()))[0]
        # pooler_output = text_encoder(text_input.input_ids.to(dist.get_device()))[1]
        # st()
        
        # load text_embedding and pooler_output to the GPU
        # text_embeddings = inp_B3HW["text_embedding"].to(dist.get_device(), non_blocking=True)
        # pooler_output = inp_B3HW["text_pooler_output"].to(dist.get_device(), non_blocking=True)

        # load image embedding
        # clip_image_embedding = inp_B3HW["image_clip_embedding"].to(dist.get_device(), non_blocking=True)
        # clip_image_pooler_output = inp_B3HW["image_clip_pooler_output"].to(dist.get_device(), non_blocking=True)
        dino_image_pooler_output = inp_B3HW["image_dino_pooler_output"].to(dist.get_device(), non_blocking=True)
        dino_image_embedding = inp_B3HW["image_dino_embedding"].to(dist.get_device(), non_blocking=True)
        # st()

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            # TODO (done): how to get logits from VAR
            # the input is input through d-layer transformer, and the output is passed a head to get the logits
            # label_B is used for generating the start of sequence token
            # here logits_BLV's shape is [B, L, 4096], the 4096 dimension output is used for predicting the word index
            # logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
            # logits_BLV = self.var(pooler_output, text_embeddings, x_BLCv_wo_first_l)
            # image condition
            # st()
            # classifier free guidance
            # logits_BLV = self.var(clip_image_pooler_output, clip_image_embedding, dino_image_embedding, x_BLCv_wo_first_l, empty_pooler_output, empty_clip_image_embedding, empty_dino_image_embedding)
            logits_BLV = self.var(pooler_output=dino_image_pooler_output, dino_condition=dino_image_embedding, x_BLCv_wo_first_l=x_BLCv_wo_first_l, empty_pooler_output = empty_pooler_output, empty_dino_embedding=empty_dino_image_embedding)
            # logits_BLV = self.var(clip_image_pooler_output, clip_image_embedding, dino_image_embedding, x_BLCv_wo_first_l)
            # TODO (done): how to train the model?
            # use CrossEntropyLoss to calculate the loss
            # In this way, they enforce the network to predict the correct word index
            # loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # st()
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) 
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        # from ipdb import set_trace as st
        # st()
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        
        # log
        # st()
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        # st()
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                # st()
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        # self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        # st()
        self.var_wo_ddp.prog_si = self.vae_local.decoder.superresolution.quantize.prog_si = -1
        return grad_norm, scale_log2
    

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        # st()
        # for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
        # FIXME: here we only load the var_wo_ddp, vae_local, not the var_opt
        for k in ('var_wo_ddp', 'vae_local'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        # st()
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
