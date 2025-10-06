set -e
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXP_NAME=$1
CONF_DIR=conf/svae

if [ -z "$PET_NPROC_PER_NODE" ]; then
    PET_NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi
torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py \
--args.load $CONF_DIR/$EXP_NAME
