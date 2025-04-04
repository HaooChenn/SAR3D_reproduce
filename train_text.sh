# Required arguments
NPROC_PER_NODE=$1  # Number of processes per node
VQVAE_PATH=$2      # Path to VQVAE checkpoint
OUT_DIR=$3         # Output directory path
DATA_DIR=$4        # Dataset directory path

# Check if all required arguments are provided
if [ -z "$NPROC_PER_NODE" ] || [ -z "$VQVAE_PATH" ] || [ -z "$OUT_DIR" ] || [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 NPROC_PER_NODE VQVAE_PATH OUT_DIR DATA_DIR"
    exit 1
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=1 --rdzv-endpoint=localhost:3000 \
    --rdzv_backend=c10d train.py \
  --depth=24 --bs=72 --ep=150 --fp16=2 --alng=1e-4 --wpe=0.1 \
  --vqvae_pretrained_path $VQVAE_PATH \
  --local_out_dir_path $OUT_DIR \
  --data_dir $DATA_DIR \
  --tblr 1e-4 \
  --num_workers 0 \
  --text_conditioned True \