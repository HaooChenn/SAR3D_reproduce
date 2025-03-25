# /mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2978 --rdzv_backend=c10d test_image.py \
#   --depth=16 --bs=1 --ep=200 --fp16=1 --alng=1e-4 --wpe=0.1 --data_path=None \
#   --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0350000.pt' \
#   --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/sample_image_condition_depth_16 \
#   --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
#   --tblr 8e-5

# nerf version
/mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2978 --rdzv_backend=c10d test_image_eval.py \
  --depth=24 --bs=1 --ep=200 --fp16=2 --alng=1e-4 --wpe=0.1 --data_path=None \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0400000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_new_from_aliyun \
  --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
  --tblr 8e-5 \
  --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_nerf_57 \
  --flexicubes False
# mesh version
# /mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2998 --rdzv_backend=c10d test_image_eval.py \
#   --depth=24 --bs=1 --ep=200 --fp16=2 --alng=1e-4 --wpe=0.1 --data_path=None \
#   --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0010000.pt' \
#   --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_new_from_aliyun \
#   --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
#   --tblr 8e-5 \
#   --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_mesh_57 \
#   --flexicubes True