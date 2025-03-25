srun -p a6000_xgpan -n 1 --gres=gpu:4 --cpus-per-task=64 -w MICL-PanXGSvr2 \
    torchrun --nproc_per_node=4 --nnodes=1 --rdzv-endpoint=MICL-PanXGSvr2:2977 --rdzv_backend=c10d train.py \
  --depth=16 --bs=60 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=/mnt/slurm_home/ywchen/data/datasets/ImageNet_subset \
  --vqvae_pretrained_path='/mnt/slurm_home/ywchen/projects/LN3Diff_VAR/logs/vae-reconstruction/objav/vae/debug2_slurm_multinode_enable_amp_scratch/model_rec1890000.pt' \
  --local_out_dir_path /mnt/slurm_home/ywchen/projects/VAR/local_output_image