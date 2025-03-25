import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from ipdb import set_trace as st
from utils import arg_util, misc
from utils.data import build_dataset, build_dataset_3D_VAR
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from datasets.g_buffer_objaverse import load_data_3D_VAR
from utils.misc import auto_resume
from ipdb import set_trace as st

# must import first here
from xformers.triton import FusedLayerNorm as LayerNorm
from xformers.components.activations import Activation
from xformers.components.feedforward import fused_mlp

from guided_diffusion import dist_util
import numpy as np
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision



def infinite_loader(loader):
    while True:
        yield from loader


def build_everything(args: arg_util.Args):
    # resume
    # st()
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # st()
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    # st()

    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # TODO: modify the vuild_vae_var function to suite my VQVAE
    # build VAE and VAR
    # st()
    from models import VAR, VQVAE, build_vae_var, build_vae_var_3D_VAR
    num_classes = 1
    vae_local, var_wo_ddp = build_vae_var_3D_VAR(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, args=args
    )
    # st()

    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        # TODO: rewrite the build_dataset function
        # num_classes, dataset_train, dataset_val = build_dataset(
        #     args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        # )
        # num_classes, dataset_train, dataset_val = build_dataset_3D_VAR(
        #     args.data_path, reso=args.data_load_reso, reso_encoder=args.reso_encoder, 
        #     preprocess=args.preprocess, load_depth=args.load_depth, imgnet_normalize=args.imgnet_normalize,
        #     dataset_size=args.dataset_size
        # )
        # # st()
        # types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
        # ld_val = DataLoader(
        #     dataset_val, num_workers=0, pin_memory=True,
        #     batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        #     shuffle=False, drop_last=False,
        # )
        # del dataset_val
        
        # ld_train = DataLoader(
        #     dataset=dataset_train, num_workers=args.workers, pin_memory=True,
        #     generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        #     batch_sampler=DistInfiniteBatchSampler(
        #         dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
        #         shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        #     ),
        # )
        # del dataset_train
        # st()
        
        ld_val = load_data_3D_VAR(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.LN3DiffConfig.image_size,
            reso_encoder=args.LN3DiffConfig.image_size_encoder,  # 224 -> 128
            num_workers=args.LN3DiffConfig.num_workers,
            load_depth=True,
            preprocess=vae_local.preprocess,  # None
            dataset_size=args.LN3DiffConfig.dataset_size,
            trainer_name=args.LN3DiffConfig.trainer_name,
            use_lmdb=args.LN3DiffConfig.use_lmdb,
            use_wds=args.LN3DiffConfig.use_wds,
            use_lmdb_compressed=args.LN3DiffConfig.use_lmdb_compressed,
            plucker_embedding=args.LN3DiffConfig.plucker_embedding,
            use_chunk=True,
            eval=True,
            load_whole=True,
            # load_depth=True # for evaluation
        )
        # st()
        ld_train = load_data_3D_VAR(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.LN3DiffConfig.image_size,
            reso_encoder=args.LN3DiffConfig.image_size_encoder,  # 224 -> 128
            num_workers=args.LN3DiffConfig.num_workers,
            load_depth=True,
            preprocess=vae_local.preprocess,  # None
            dataset_size=args.LN3DiffConfig.dataset_size,
            trainer_name=args.LN3DiffConfig.trainer_name,
            use_lmdb=args.LN3DiffConfig.use_lmdb,
            use_wds=args.LN3DiffConfig.use_wds,
            use_lmdb_compressed=args.LN3DiffConfig.use_lmdb_compressed,
            plucker_embedding=args.LN3DiffConfig.plucker_embedding,
            use_chunk=True,
            load_whole=False,
            # load_depth=True # for evaluation
        )
        # st()
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        # print("training_dataset_size", len(ld_train))
        # debug
        iters_train = int(125 / args.batch_size)
        # train
        # iters_train = int(12500 / args.batch_size)
        print("iters_train", iters_train)
        # st()
        ld_train = infinite_loader(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        # print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var, build_vae_var_3D_VAR
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    # TODO: pls see the following function
    # st()
    # vae_local, var_wo_ddp = build_vae_var(
    #     V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
    #     device=dist.get_device(), patch_nums=args.patch_nums,
    #     num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
    #     flash_if_available=args.fuse, fused_if_available=args.fuse,
    #     init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    # )


    
    # TODO: change the path to the correct path, given args.local_out_dir_path
    # vae_ckpt = 'vae_ch160v4096z32.pth'
    # st()
    vae_ckpt = args.vqvae_pretrained_path
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    # st()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    # st()
    # here args.vfast=args.tfast=0, so the model will NOT be compiled
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    # st()
    # print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.decoder.superresolution.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    # st()
    # build optimizer
    # filter the parameters that need to be calculated in weight decay loss
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    # st()
    # args.afuse is False here
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    # args.fp16 is 1 here
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    # from tqdm import tqdm
    # from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
    # tokenizer = CLIPTokenizer.from_pretrained("huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
    # text_encoder = CLIPTextModel.from_pretrained("huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
    # clipmodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    # processer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # Dinov2
    # from transformers import AutoImageProcessor, Dinov2Model
    # dino_image_processor = AutoImageProcessor.from_pretrained("huggingface/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
    # dino_model = Dinov2Model.from_pretrained("huggingface/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
    # trainer = VARTrainer(
    #     device=args.device, patch_nums=args.patch_nums, resos=args.resos,
    #     vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
    #     var_opt=var_optim, label_smooth=args.ls,
    #     dino_image_model=dino_model,
    #     dino_image_processor=dino_image_processor,
    # )
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
        dino_image_model=None,
        dino_image_processor=None,
    )
    # # 检查优化器的状态
    # for state in  trainer.var_opt.optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == 'step':
    #                 print(f'[optimizer state] {k} {v}')
    #             if v.device == 'cpu':
    #                 print(f'[optimizer state] {k} {v} {v.device}')
    # st()
    # trainer.var_opt.load_state_dict(trainer_state['var_opt'], strict=True)
    # for state in trainer.var_opt.optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == 'step':
    #                 print(f'[optimizer state] {k} {v}')
    #             if v.device == 'cpu':
    #                 print(f'[optimizer state] {k} {v} {v.device}')
    # st()
    if trainer_state is not None and len(trainer_state):
        # trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
        trainer.load_state_dict(trainer_state, strict=True, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    # for state in trainer.var_opt.optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == 'step':
    #                 print(f'[optimizer state] {k} {v}')
    #             if v.device == 'cpu':
    #                 print(f'[optimizer state] {k} {v} {v.device}')
    #                 del state[k]
    # st()
    # for state in trainer.var_opt.optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == 'step':
    #                 print(f'[optimizer state] {k} {v}')
    #             if v.device == 'cpu':
    #                 print(f'[optimizer state] {k} {v} {v.device}')
    # st()
    # for state in trainer.var_opt.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             if k == 'step':
    #                 print(f'[optimizer state] {k} {v}')
    #             if v.device == 'cpu':
    #                 print(f'[optimizer state] {k} {v} {v.device}')
    # st()
    if args.local_debug:
        rng = torch.Generator('cpu')
        # rng.manual_seed(0)
        rng.manual_seed(3407)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )

@torch.inference_mode()
def render_video_given_triplane(planes,
                                rec_model,
                                name_prefix='0',
                                save_img=False,
                                render_reference=None,
                                save_mesh=False,
                                save_path="./sample_save",):
    pool_128 = torch.nn.AdaptiveAvgPool2d((128, 128))
    pool_512 = torch.nn.AdaptiveAvgPool2d((512, 512))

    batch_size = planes.shape[0]

    ddpm_latent = {'latent_after_vit': planes.to(torch.float32)}
    # st()


    if save_mesh: # ! tune marching cube grid size according to the vram size
        mesh_size = 192

        mesh_thres = 10  # TODO, requires tuning
        import mcubes
        import trimesh
        os.makedirs(save_path, exist_ok=True)
        dump_path = f'{save_path}/mesh/'
        os.makedirs(dump_path, exist_ok=True)
        # st()
        grid_out = rec_model(
            latent=ddpm_latent,
            grid_size=mesh_size,
            behaviour='triplane_decode_grid',
        )
        # st()
        vtx, faces = mcubes.marching_cubes(
            grid_out['sigma'].to(torch.float32).squeeze(0).squeeze(-1).cpu().numpy(),
            mesh_thres)
        # st()
        grid_scale = [rec_model.decoder.rendering_kwargs['sampler_bbox_min'], rec_model.decoder.rendering_kwargs['sampler_bbox_max']]
        vtx = (vtx / (mesh_size-1) * 2 - 1 ) * grid_scale[1] # normalize to g-buffer objav dataset scale

        # ! save normalized color to the vertex
        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=dist_util.dev()).unsqueeze(0)
        # st()
        vtx_colors = rec_model.decoder.forward_points(ddpm_latent['latent_after_vit'], vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors.clip(0,1) * 255).astype(np.uint8)

        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
        mesh.export(mesh_dump_path, 'ply')

        print(f"Mesh dumped to {dump_path}")
        del grid_out, mesh
        torch.cuda.empty_cache()
        # st()
        # return

    video_out = imageio.get_writer(
        f'{save_path}/triplane_{name_prefix}.mp4',
        mode='I',
        fps=15,
        codec='libx264')

    if render_reference is None:
        raise ValueError('render_reference is None')
    else:  # use train_traj
        for key in ['ins', 'bbox', 'caption']:
            if key in render_reference:
                render_reference.pop(key)

        # compat lst for enumerate
        render_reference = [{
            k: v[idx:idx + 1]
            for k, v in render_reference.items()
        } for idx in range(40)]

    # for i, batch in enumerate(tqdm(self.eval_data)):
    # st()
    for i, batch in enumerate(tqdm(render_reference)):
        micro = {
            k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        pred = rec_model(
            latent={
                'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)
            },
            c=micro['c'],
            behaviour='triplane_dec')
        # st()
        pred_depth = pred['image_depth']
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                        pred_depth.min())

        # save viridis_r depth
        pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
        pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
        pred_depth = torch.from_numpy(pred_depth).to(
            pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)
        # False here
        if 'image_sr' in pred:

            gen_img = pred['image_sr']

            if pred['image_sr'].shape[-1] == 512:

                pred_vis = torch.cat([
                    micro['img_sr'],
                    pool_512(pred['image_raw']), gen_img,
                    pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                    dim=-1)

            elif pred['image_sr'].shape[-1] == 128:

                pred_vis = torch.cat([
                    micro['img_sr'],
                    pool_128(pred['image_raw']), pred['image_sr'],
                    pool_128(pred_depth).repeat_interleave(3, dim=1)
                ],
                                    dim=-1)

        else:
            gen_img = pred['image_raw']

            pred_vis = torch.cat(
                [
                    # self.pool_128(micro['img']),
                    # pool_128(gen_img),
                    # pool_512(gen_img),
                    gen_img,
                    # self.pool_128(pred_depth.repeat_interleave(3, dim=1))
                    # pool_128(pred_depth)
                    # pool_512(pred_depth)
                    pred_depth
                ],
                dim=-1)  # B, 3, H, W

        if save_img:
            for batch_idx in range(gen_img.shape[0]):
                sampled_img = Image.fromarray(
                    (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                        127.5 + 127.5).clip(0, 255).astype(np.uint8))
                if sampled_img.size != (512, 512):
                    sampled_img = sampled_img.resize(
                        (128, 128), Image.HAMMING)  # for shapenet
                sampled_img.save(save_path +
                                    '/FID_Cals/{}_{}.png'.format(
                                        int(name_prefix) * batch_size +
                                        batch_idx, i))
                # print('FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))

        vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
        vis = vis * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        for j in range(vis.shape[0]
                        ):  # ! currently only export one plane at a time
            video_out.append_data(vis[j])

    # if not save_img:
    video_out.close()
    del video_out
    print('logged video to: ',
            f'{save_path}/triplane_{name_prefix}.mp4')

    del vis, pred_vis, micro, pred,

@torch.inference_mode()
def log_training_samples_txt(planes,
                                rec_model,
                                name_prefix='0',
                                save_img=False,
                                render_reference=None,
                                save_mesh=False,
                                save_path="./sample_save",
                                caption=None,
                                gt_img=None):
    pool_128 = torch.nn.AdaptiveAvgPool2d((128, 128))
    pool_512 = torch.nn.AdaptiveAvgPool2d((512, 512))

    batch_size = planes.shape[0]

    ddpm_latent = {'latent_after_vit': planes.to(torch.float32)}
    # st()


    # if save_mesh: # ! tune marching cube grid size according to the vram size
        # mesh_size = 192

        # mesh_thres = 10  # TODO, requires tuning
        # import mcubes
        # import trimesh
        # os.makedirs(save_path, exist_ok=True)
        # dump_path = f'{save_path}/mesh/'
        # os.makedirs(dump_path, exist_ok=True)
        # # st()
        # grid_out = rec_model(
        #     latent=ddpm_latent,
        #     grid_size=mesh_size,
        #     behaviour='triplane_decode_grid',
        # )
        # # st()
        # vtx, faces = mcubes.marching_cubes(
        #     grid_out['sigma'].to(torch.float32).squeeze(0).squeeze(-1).cpu().numpy(),
        #     mesh_thres)
        # # st()
        # grid_scale = [rec_model.decoder.rendering_kwargs['sampler_bbox_min'], rec_model.decoder.rendering_kwargs['sampler_bbox_max']]
        # vtx = (vtx / (mesh_size-1) * 2 - 1 ) * grid_scale[1] # normalize to g-buffer objav dataset scale

        # # ! save normalized color to the vertex
        # vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=dist_util.dev()).unsqueeze(0)
        # # st()
        # vtx_colors = rec_model.decoder.forward_points(ddpm_latent['latent_after_vit'], vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        # vtx_colors = (vtx_colors.clip(0,1) * 255).astype(np.uint8)

        # mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # mesh_dump_path = os.path.join(dump_path, f'{name_prefix}.ply')
        # mesh.export(mesh_dump_path, 'ply')

        # print(f"Mesh dumped to {dump_path}")
        # del grid_out, mesh
        # torch.cuda.empty_cache()
        # st()
        # return

    # video_out = imageio.get_writer(
    #     f'{save_path}/triplane_{name_prefix}.mp4',
    #     mode='I',
    #     fps=15,
    #     codec='libx264')

    # if render_reference is None:
    #     raise ValueError('render_reference is None')
    # else:  # use train_traj
    #     for key in ['ins', 'bbox', 'caption']:
    #         if key in render_reference:
    #             render_reference.pop(key)

    #     # compat lst for enumerate
    #     render_reference = [{
    #         k: v[idx:idx + 1]
    #         for k, v in render_reference.items()
    #     } for idx in range(12)]
    # st()
    # for i, batch in enumerate(tqdm(self.eval_data)):
    # st()
    pred = rec_model(
            latent={
                'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)
            },
            c=render_reference['c'],
            behaviour='triplane_dec')
    save_img = torch.cat([gt_img.to(pred['image_raw'].device), pred['image_raw']], dim=3)
    # st()
    # for i, batch in enumerate(tqdm(render_reference)):
    #     micro = {
    #         k: v.to(dist_util.dev()) if isinstance(v, torch.Tensor) else v
    #         for k, v in batch.items()
    #     }
    #     pred = rec_model(
    #         latent={
    #             'latent_after_vit': ddpm_latent['latent_after_vit'].repeat_interleave(6, dim=0).repeat(2,1,1,1)
    #         },
    #         c=micro['c'],
    #         behaviour='triplane_dec')
    #     # st()
    #     # pred_depth = pred['image_depth']
    #     # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
    #     #                                                 pred_depth.min())

    #     # # save viridis_r depth
    #     # pred_depth = pred_depth.cpu()[0].permute(1, 2, 0).numpy()
    #     # pred_depth = (plt.cm.viridis(pred_depth[..., 0])[..., :3]) * 2 - 1
    #     # pred_depth = torch.from_numpy(pred_depth).to(
    #     #     pred['image_raw'].device).permute(2, 0, 1).unsqueeze(0)
    #     # False here
    #     if 'image_sr' in pred:

    #         gen_img = pred['image_sr']

    #         if pred['image_sr'].shape[-1] == 512:

    #             pred_vis = torch.cat([
    #                 micro['img_sr'],
    #                 pool_512(pred['image_raw']), gen_img,
    #                 # pool_512(pred_depth).repeat_interleave(3, dim=1)
    #             ],
    #                                 dim=-1)

    #         elif pred['image_sr'].shape[-1] == 128:

    #             pred_vis = torch.cat([
    #                 micro['img_sr'],
    #                 pool_128(pred['image_raw']), pred['image_sr'],
    #                 # pool_128(pred_depth).repeat_interleave(3, dim=1)
    #             ],
    #                                 dim=-1)

    #     else:
    #         gen_img = pred['image_raw']

            # pred_vis = torch.cat(
            #     [
            #         # self.pool_128(micro['img']),
            #         pool_128(gen_img),
            #         # self.pool_128(pred_depth.repeat_interleave(3, dim=1))
            #         pool_128(pred_depth)
            #     ],
            #     dim=-1)  # B, 3, H, W
        # st()
        # if save_img:
        #     for batch_idx in range(gen_img.shape[0]):
        #         sampled_img = Image.fromarray(
        #             (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
        #                 127.5 + 127.5).clip(0, 255).astype(np.uint8))
        #         if sampled_img.size != (512, 512):
        #             sampled_img = sampled_img.resize(
        #                 (128, 128), Image.HAMMING)  # for shapenet
        #         sampled_img.save(save_path +
        #                             '/FID_Cals/{}_{}.png'.format(
        #                                 int(name_prefix) * batch_size +
        #                                 batch_idx, i))
                # print('FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))

        # vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
        # vis = vis * 127.5 + 127.5
        # vis = vis.clip(0, 255).astype(np.uint8)
        # for j in range(vis.shape[0]
        #                 ):  # ! currently only export one plane at a time
        #     video_out.append_data(vis[j])

    # if not save_img:
    # video_out.close()
    # del video_out
    # print('logged video to: ',
    #         f'{save_path}/triplane_{name_prefix}.mp4')

    # del vis, pred_vis, micro, pred,
    del pred_vis, micro, pred,

def main_training():
    # TODO: remove the following line
    # torch.autograd.set_detect_anomaly(True)
    # st()

    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    # st()
    # for ep in range(start_ep, args.ep):
    #     # st()
    #     if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
    #         ld_train.sampler.set_epoch(ep)
    #         if ep < 3:
    #             # noinspection PyArgumentList
    #             print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
    #     tb_lg.set_step(ep * iters_train)
    it = 0
    for data in ld_val:
        B, V = data["image_dino_embedding"].shape[0], trainer.vae_local.decoder.superresolution.quantize.vocab_size
        pooler_output = data["image_dino_pooler_output"].to(dist.get_device(), non_blocking=True)
        dino_embeddings = data["image_dino_embedding"].to(dist.get_device(), non_blocking=True)
        caption = data["single_image"]
        # st()



        # =====================generate condition online================================ #
        from PIL import Image
        single_image = Image.open("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/sample_test_image/genshin_teapot-input.png")
        background = Image.new("RGBA", np.array(single_image).shape[0:2], (255, 255, 255, 255))
        background.paste(single_image, (0, 0), single_image)
        resized_img = background.resize((256, 256), Image.ANTIALIAS)
        final_img = resized_img.convert('RGB')
        final_img.save('resized_image.png')
        single_image = Image.open('resized_image.png')
        # st()
        from transformers import AutoImageProcessor, Dinov2Model
        dino_image_processor = AutoImageProcessor.from_pretrained("huggingface/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
        dino_model = Dinov2Model.from_pretrained("huggingface/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
        with torch.no_grad():
            dino_image = dino_image_processor(images=single_image, return_tensors="pt")
            dinomodel_output = dino_model(**dino_image)
            dino_embeddings = dinomodel_output.last_hidden_state[:,1:,:].to(dist.get_device(), non_blocking=True)
            pooler_output = dinomodel_output.pooler_output.to(dist.get_device(), non_blocking=True)
        # =====================generate condition online================================ #

        with torch.inference_mode():
            triplane = trainer.var_wo_ddp.autoregressive_infer_cfg_3D_VAR_image_l2norm(B=B, dino_image_embeddings=dino_embeddings, pooler_output=pooler_output, cfg=4, top_k=900, top_p=0.95, more_smooth=False)
        camera = torch.empty(40, 25, device=args.device)
        camera = torch.load("./camera.pt")
        #FIXME: find out why the camera is not correct
        rot = torch.tensor([[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ], dtype=torch.float32, device=camera.device)
        for i in range(camera.shape[0]):
            camera[i][0:16] = torch.mm(rot,camera[i][0:16].reshape(4,4)).reshape(-1)
        for i in range(triplane.shape[0]):
            triplane_i = triplane[i].unsqueeze(0)
            render_video_given_triplane(triplane_i, trainer.vae_local, name_prefix=f'{it}', render_reference={'c': camera}, save_img=False, save_mesh=True, save_path=args.local_out_dir_path)
            from PIL import Image
            Image.fromarray(caption[i].numpy().astype(np.uint8)).save(args.local_out_dir_path + f"/input_{it}.png")
            it += 1
        
        # st()





    #     with torch.inference_mode():
    #         for i in range(min(triplane.shape[0],4)):
    #             triplane_i = triplane[i].unsqueeze(0)
    #             camera = data_eval['nv_c'][12*i:12*(i+1)].to(dist.get_rank())
    #             pred = trainer.vae_local(
    #                 latent={
    #                     'latent_after_vit': triplane_i.to(torch.float32).repeat_interleave(6, dim=0).repeat(2,1,1,1)
    #                 },
    #                 c=camera,
    #                 behaviour='triplane_dec')
    #             gt_img = data_eval['nv_img'][12*i:12*(i+1)]
    #             save_img_i = torch.cat([pred['image_raw'][0:6], gt_img[0:6].to(pred['image_raw'].device)], dim=2)
    #             save_img.append(save_img_i)
    #     save_img = torch.cat(save_img, dim=2)

















        
    #     L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
    #     best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
    #     if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
    #     args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
    #     args.cur_ep = f'{ep+1}/{args.ep}'
    #     args.remain_time, args.finish_time = remain_time, finish_time
        
    #     AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
    #     # is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
    #     is_val_and_also_saving = 1
    #     if is_val_and_also_saving:
    #         # TODO: write eval_ep here
    #         # val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
    #         # best_updated = best_val_loss_tail > val_loss_tail
    #         # best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
    #         # best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
    #         # AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
    #         # args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
    #         # print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
    #         best_updated = False
    #         if dist.is_local_master():
    #             # # validation
    #             local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-{}.pth'.format(ep))
    #             local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
    #             print(f'[saving ckpt] ...', end='', flush=True)
    #             torch.save({
    #                 'epoch':    ep+1,
    #                 'iter':     (ep+1) * iters_train,
    #                 'trainer':  trainer.state_dict(),
    #                 'args':     args.state_dict(),
    #             }, local_out_ckpt)
    #             if best_updated:
    #                 shutil.copy(local_out_ckpt, local_out_ckpt_best)
    #             print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
    #             # triplane, caption, data_eval = trainer.eval_ep_3D_VAR(ld_train)
    #             triplane, caption, data_eval = trainer.eval_ep_3D_VAR(ld_val)
    #             save_img = []
    #             with torch.inference_mode():
    #                 for i in range(min(triplane.shape[0],4)):
    #                     triplane_i = triplane[i].unsqueeze(0)
    #                     camera = data_eval['nv_c'][12*i:12*(i+1)].to(dist.get_rank())
    #                     pred = trainer.vae_local(
    #                         latent={
    #                             'latent_after_vit': triplane_i.to(torch.float32).repeat_interleave(6, dim=0).repeat(2,1,1,1)
    #                         },
    #                         c=camera,
    #                         behaviour='triplane_dec')
    #                     gt_img = data_eval['nv_img'][12*i:12*(i+1)]
    #                     save_img_i = torch.cat([pred['image_raw'][0:6], gt_img[0:6].to(pred['image_raw'].device)], dim=2)
    #                     save_img.append(save_img_i)
    #             save_img = torch.cat(save_img, dim=2)
    #             os.makedirs(os.path.join(args.local_out_dir_path, 'rendered'), exist_ok=True)
    #             torchvision.utils.save_image(save_img, os.path.join(args.local_out_dir_path, 'rendered', f'ep_{ep}.png'), nrow=6)


                
    #             del triplane, caption, data_eval, save_img, camera, pred, gt_img, save_img_i, local_out_ckpt, local_out_ckpt_best, best_updated
    #         dist.barrier()
        
    #     print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
    #     tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
    #     tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
    #     args.dump_log(); tb_lg.flush()
    
    # total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    # print('\n\n')
    # print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    # print('\n\n')
    
    # del stats
    # del iters_train, ld_train
    # time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    # args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    # print(f'final args:\n\n{str(args)}')
    # args.dump_log(); tb_lg.flush(); tb_lg.close()
    # dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    # st()
    step_cnt = 0
    # st()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    # for it, (data) in  me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
    #     if 'a blue and pink polka dot shark chair.'in data['caption'] or \
    #         'a blue parrot in flight.' in data['caption'] or \
    #         'a Minecraft-inspired spider-scorpion hybrid with orange and brown legs, made from Lego bricks.' in data['caption'] or \
    #         'a white horned Totoro-egg hybrid with owl-like features and hat ears.' in data['caption']:
    #         print(data['caption'], data['img_path'])
    #     st()
    
    
    # TODO:copy the code from LN3Diff to iterate the dataloader and get the input and label
    # for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
    # st()
    # for data in enumerate(ld_or_itrt):
    #     st()
    for it, (data) in  me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        # print("it:", it)
        # st()
        # TODO:copy the code from LN3Diff to iterate the dataloader, get the feature map of the input image
        # print(data['caption'])
        # st()
        # inp = data['img_to_encoder']
        

        # FIXME: the label is fixed to 0 here
        # label = torch.ones(int(inp.size(0)/12), dtype=torch.int64)
        label = torch.ones(args.batch_size, dtype=torch.int64)


        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        # inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        # wp_it = args.wp * iters_train
        # FIXME: hard code the wp_it here
        wp_it = 5000
        # st()
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        # False here
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        # st()
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        # st()

        # st()

        # FIXME: please delete the following code after change the dataloader
        # indices = torch.cat([torch.arange(i, i + 6) for i in range(0, inp.size(0), 12)])
        # inp = inp[indices]
        # st()


        # grad_norm, scale_log2 = trainer.train_step(
        #     it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        # )
        # classifier free guidance
        # empty_clip_image_embedding = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_embedding.npy")).to(data['image_clip_pooler_output'].device).unsqueeze(0)
        # empty_pooler_output = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_clip_pooler_output.npy")).to(data['image_clip_pooler_output'].device).unsqueeze(0)
        # empty_dino_image_embedding = torch.from_numpy(np.load("/mnt/slurm_home/ywchen/projects/VAR-image/VAR/empty_dino_embedding.npy"))[1:, :].to(data['image_clip_pooler_output'].device).unsqueeze(0)
        empty_pooler_output = torch.from_numpy(np.load("./empty_dino_pooler_output.npy")).to(dist.get_device()).unsqueeze(0)
        empty_dino_image_embedding = torch.from_numpy(np.load("./empty_dino_embedding.npy"))[1:, :].to(dist.get_device()).unsqueeze(0)
        # empty_pooler_output = torch.from_numpy(np.load("./empty_dino_pooler_output.npy")).to(data['image_dino_pooler_output'].device).unsqueeze(0)
        # empty_dino_image_embedding = torch.from_numpy(np.load("./empty_dino_embedding.npy"))[1:, :].to(data['image_dino_pooler_output'].device).unsqueeze(0)
        # grad_norm, scale_log2 = trainer.train_step(
        #     it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        #     empty_pooler_output=empty_pooler_output, empty_clip_image_embedding=empty_clip_image_embedding,
        #     empty_dino_image_embedding = empty_dino_image_embedding
        # )
        from trainer import VARTrainer
        VARTrainer.train_step
        # grad_norm, scale_log2 = trainer.train_step(
        #     it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        #     empty_pooler_output=empty_pooler_output,
        #     empty_dino_image_embedding = empty_dino_image_embedding
        # )
        import time
        start = time.time()
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
            empty_pooler_output=empty_pooler_output,
            empty_dino_image_embedding = empty_dino_image_embedding,
            # VAE_model = trainer.vae_local,
            # quantize_model = trainer.quantize_local,
        )
        print("time:", time.time() - start)
        # grad_norm, scale_log2 = trainer.train_step(
        #     it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
        #     inp_B3HW=data, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        # )
        # st()
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
