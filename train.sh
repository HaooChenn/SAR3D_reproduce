
# d16, 256x256
# debug
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --rdzv-endpoint=MICL-PanXGSvr2:2978 --rdzv_backend=c10d train.py \
  --depth=16 --bs=4 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/VAR-image/VAR/log/LN3Diff/model_rec0910000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_debug

# train l2norm
torchrun --nproc_per_node=2 --nnodes=1 --rdzv-endpoint=localhost:2977 --rdzv_backend=c10d train.py \
  --depth=16 --bs=26 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/VAR-i mage/VAR/log/LN3Diff/model_rec0910000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/log/resume_2_cards

/usr/local/slurm20/bin/srun -p a6000_xgpan -w MICL-PanXGSvr2 --job-name=test --gres=gpu:2 --kill-on-bad-exit=1 --pty bash -i

# sample script
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=MICL-PanXGSvr2:2977 --rdzv_backend=c10d sample_LN3Diff.py \
  --depth=16 --bs=12 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/logs/vae-reconstruction/objav/vae/debug2_slurm_multinode_enable_amp_scratch/model_rec1890000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_debug3

# sample script text
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=MICL-PanXGSvr2:2977 --rdzv_backend=c10d sample_LN3Diff_text.py \
  --depth=16 --bs=12 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/logs/vae-reconstruction/objav/vae/debug2_slurm_multinode_enable_amp_scratch/model_rec1890000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_debug3


# produce latent
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2999 --rdzv_backend=c10d produce_latent.py \
  --depth=16 --bs=1 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset   \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/logs/vae-reconstruction/objav/vae/debug2_slurm_multinode_enable_amp_scratch/model_rec1890000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_test

# produce latent l2norm
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2932 --rdzv_backend=c10d produce_latent_l2norm.py \
  --depth=16 --bs=1 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset   \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/VAR-image/VAR/log/LN3Diff/model_rec0910000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_test

#srun
srun -p a6000_xgpan -n 1 --gres=gpu:2 --cpus-per-task=32 -w MICL-PanXGSvr2 torchrun --nproc_per_node=2 --nnodes=1 --rdzv-endpoint=localhost:2977 --rdzv_backend=c10d train.py \
  --depth=16 --bs=26 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/VAR-image/VAR/log/LN3Diff/model_rec0910000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/log/resume_2_cards