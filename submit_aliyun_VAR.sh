conda activate py39_chen
torchrun --nproc_per_node=7 --nnodes=1 --rdzv-endpoint=localhost:3000 --rdzv_backend=c10d train.py \
  --depth=24 --bs=63 --ep=10000 --fp16=2 --alng=1e-4 --wpe=0.1 --data_path=None \
  --vqvae_pretrained_path='/nas/shared/public/yslan/cyw/ln3diff_VAR/log/seperate_GAN_LRM_resume/model_rec0350000.pt' \
  --local_out_dir_path /nas/shared/public/yslan/cyw/ln3diff_VAR/log/VAR_Image/local_output_test_offline_latent_depth_24_bf16 \
  --data_dir /cpfs01/user/lanyushi.p/data/chunk-png-normal/bs_12_shuffle/170K/256/ \
  --tblr 1e-4