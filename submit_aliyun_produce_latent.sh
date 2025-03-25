conda activate py39_chen
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-endpoint=localhost:2932 --rdzv_backend=c10d produce_latent_l2norm.py \
  --depth=16 --bs=1 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --data_path=None   \
  --vqvae_pretrained_path='/nas/shared/public/yslan/cyw/ln3diff_VAR/log/seperate_GAN_LRM_resume/model_rec0350000.pt' \
  --local_out_dir_path /nas/shared/public/yslan/cyw/ln3diff_VAR/log/VAR_Image/local_output_test \
  --data_dir /cpfs01/user/lanyushi.p/data/chunk-png-normal-latent/bs_12_shuffle/170K/256/