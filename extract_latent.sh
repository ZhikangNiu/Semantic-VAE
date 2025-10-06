#!/bin/bash
set -e
export OMP_NUM_THREADS=4
exp=$1
nnodes=1
nproc_per_node=8

tag=600k
ckpt_name="ckpts/${exp}"
output_base="ldm_features/${exp}_ema/"
use_ema=True

torchrun --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  scripts/extract_latent.py \
  --ckpt_dir $ckpt_name \
  --input_dir /inspire/dataset/librispeech/v1/test-clean/ \
  --output_dir "$output_base/LibriSpeech/test-clean" \
  --model_tag $tag \
  --use_ema