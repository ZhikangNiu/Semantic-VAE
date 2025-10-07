#!/bin/bash
set -e
export OMP_NUM_THREADS=4

nnodes=1
nproc_per_node=8
tag=600k
ckpt_path=YOUR_EXP_TOP_ROOT
npy_root=LATENT_ROOT

torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
scripts/recon_latent_to_wave.py \
--ckpt_dir $ckpt_path \
--input_dir $npy_root \
--model_tag $tag \
--use_ema
