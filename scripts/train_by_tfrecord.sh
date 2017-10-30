#!/bin/bash

export CUDA_VISIBLE_DEVICES=12,13,14,15

python3 train_by_tfrecord_BN.py \
    --train_dir=ckpt/ucf/rgb/v3_BN   \
    --dataset_dir=/data2/jyli/UCF101/set1/train/tfrecord \
    --model_name=inception_v3_frozen_BN \
    --checkpoint_path=ckpt/inception_v3.ckpt  \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=3500  \
    --epoch_size=9537  \
    --num_clones=4  \
    --num_classes=101  \
    --batch_size=16  \
    --train_image_size=299 \
    --agg_fn=average \
    --topK=3  \
    --segment_num=5  \
    --learning_rate=0.001 \
    --num_steps_per_decay=1500 \
    --learning_rate_decay_factor=0.1 \
    --learning_rate_decay_type=exponential \
    --save_interval_secs=100 \
    --save_summaries_secs=100 \
    --log_every_n_steps=50 \
    --optimizer=momentum \
    --weight_decay=0.0005
