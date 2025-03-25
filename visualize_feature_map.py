################## 1. Download checkpoints and build models

# must import first here
from xformers.triton import FusedLayerNorm as LayerNorm
from xformers.components.activations import Activation
from xformers.components.feedforward import fused_mlp

import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var_3D_VAR, build_vae_var
from ipdb import set_trace as st
from guided_diffusion import dist_util
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


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
        dump_path = f'./{save_path}/mesh/'
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
                    pool_128(gen_img),
                    # self.pool_128(pred_depth.repeat_interleave(3, dim=1))
                    pool_128(pred_depth)
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

def visualize_feature_map_average(feature_maps, save_path):
    # 平均所有通道
    average_map = np.mean(feature_maps, axis=0)
    
    # 绘制图像
    plt.imsave(save_path, average_map, cmap='viridis')

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

var_ckpt = f'/mnt/slurm_home/ywchen/projects/VAR/local_output_use_half_data/ar-ckpt-102.pth'

# download checkpoint
# hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
# vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
vae_ckpt = '/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/logs/vae-reconstruction/objav/vae/debug2_slurm_multinode_enable_amp_scratch/model_rec1500000.pt'
# if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
# if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    num_classes = 1
    import dist
    from utils import arg_util
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    vae, var = build_vae_var_3D_VAR(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=patch_nums,
        num_classes=num_classes, depth=MODEL_DEPTH, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, args=args
    )
# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu')['trainer']['var_wo_ddp'], strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')
# st()
############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}

# class_labels = np.random.randint(0, 1000, 100)  #@param {type:"raw"}
class_labels = torch.ones(50, dtype=torch.long)
print(f'class_labels: {class_labels}')
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
with torch.inference_mode():
    # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
    #     # TODO: sample 3d aseets here, output video
    #     triplane = var.autoregressive_infer_cfg_3D_VAR(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
    triplane, featue_map = var.autoregressive_output_feature(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
    for i in range(B):
        for j in range(10):
            if j == 0:
                featue_map_0 = featue_map[0][j][i].squeeze(0)
                featue_map_1 = featue_map[1][j][i].squeeze(0)
                featue_map_2 = featue_map[2][j][i].squeeze(0)
            else:
                featue_map_0 = torch.cat((featue_map_0, featue_map[0][j][i].squeeze(0)), 1)
                featue_map_1 = torch.cat((featue_map_1, featue_map[1][j][i].squeeze(0)), 1)
                featue_map_2 = torch.cat((featue_map_2, featue_map[2][j][i].squeeze(0)), 1)
        visualize_feature_map_average(featue_map_0.cpu().numpy(), "./sample_save_half_data/feature_map_0_object_{}.png".format(i))
        visualize_feature_map_average(featue_map_1.cpu().numpy(), "./sample_save_half_data/feature_map_1_object_{}.png".format(i))
        visualize_feature_map_average(featue_map_2.cpu().numpy(), "./sample_save_half_data/feature_map_2_object_{}.png".format(i))
    st()
    # gt_idx = torch.load("/mnt/slurm_home/ywchen/projects/VAR/gt_idx_Bl.pth")
    # triplane = var.reconstruct_gt_Bl_idx(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth, gt_idx=gt_idx)
    camera = torch.empty(40, 25, device=device)
    camera = torch.load("/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/camera.pt")

    for i in range(triplane.shape[0]):
        triplane_i = triplane[i].unsqueeze(0)
        render_video_given_triplane(triplane_i, vae, name_prefix=f'{i}', render_reference={'c': camera}, save_img=False, save_mesh=True, save_path="./sample_save_half_data")



