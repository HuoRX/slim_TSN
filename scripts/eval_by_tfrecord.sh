#!/bin/bash

# predict one frame by using predict_frame_script.py
# It's the same as slim/predict_frame.py

export CUDA_VISIBLE_DEVICES=12,13,14,15

python3 eval_by_tfrecord.py \
    --dataset_dir=/data2/jyli/UCF101/set1/test/tfrecord \
    --model_name=inception_v3_frozen_BN \
    --checkpoint_path=ckpt/ucf/rgb/v3_BN/allvar \
    --batch_size=2  \
    --num_classes=101  \
    --segment_num=25  \
    --agg_fn=average \
    --sample_size=3800  \
    --recall_at=2  \
    --num_length=1 \
    --num_preprocessing_threads=4 
