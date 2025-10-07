export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

nproc_per_node=8

SSL_MODEL_PATH=facebook/wavlm-large/ # or local path

OUTPUT_ROOT=../ssl_features/wavlm-features/ # output directory
mkdir -p $OUTPUT_DIR

AUDIO_ROOT_DIR=YOUR_PATH/libritts/v1/

for subset in dev-clean;do
    AUDIO_ROOT=$AUDIO_ROOT_DIR/$subset
    OUTPUT_DIR=$OUTPUT_ROOT/$subset
    echo "Process $AUDIO_ROOT"
    torchrun \
    --nnodes 1 \
    --nproc_per_node $nproc_per_node \
    extract_ssl_feature.py \
    --audio-root-dir $AUDIO_ROOT \
    --output-dir     $OUTPUT_DIR \
    --ssl-model-path $SSL_MODEL_PATH \
    --file-extension .wav \
    --layer-index -1 \
    --num-workers 16 \
    --log-every 500
done
