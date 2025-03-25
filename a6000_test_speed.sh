conda activate py39
/mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2978 --rdzv_backend=c10d test_image_test_speed.py \
  --depth=24 --bs=1 --ep=200 --fp16=2 --alng=1e-4 --wpe=0.1 --data_path=None \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0010000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_mesh_and_nerf \
  --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
  --tblr 8e-5 \
  --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_wild_speed_mesh \
  --flexicubes True

# /mnt/slurm_home/ywchen/anaconda3/envs/py39/bin/torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2979 --rdzv_backend=c10d test_image_test_speed.py \
#   --depth=24 --bs=1 --ep=200 --fp16=2 --alng=1e-4 --wpe=0.1 --data_path=None \
#   --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_Ori/LN3diff_VAR/logs/vae-reconstruction/objav/vae/pretrained/model_rec0400000.pt' \
#   --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/local_output_mesh_and_nerf \
#   --data_dir /mnt/slurm_home/ywchen/data/datasets/objv_chunk_v=6/bs_12_shuffle/170K/256 \
#   --tblr 8e-5 \
#   --save_path /mnt/slurm_home/ywchen/projects/VAR-image/VAR/eval_wild_speed_nerf \
#   --flexicubes False