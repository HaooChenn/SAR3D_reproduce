# # mesh version
# /mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2932 \
#   --rdzv_backend=c10d produce_latent_l2norm.py \
#   --depth=24 --bs=1 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=None   \
#   --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0010000.pt' \
#   --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_mesh_and_nerf \
#   --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
#   --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/sample_256_mesh \
#   --flexicubes True

# nerf version
# conda activate py39
/mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2932 \
  --rdzv_backend=c10d produce_latent_l2norm.py \
  --depth=16 --bs=1 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=None   \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0400000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_test \
  --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
  --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/sample_256_nerf \
  --flexicubes False
  