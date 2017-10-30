#!/bin/bash

# The commands below fine tune the pretrained VGG16_slim 

# DATASET_DIR = train data location
# TRAIN_DIR = where the new checkpoint save
# CHECKPOINT_PATH = net/VGG_slim/vgg_16.ckpt  checkpoint file for pretrain net
#    --checkpoint_exclude_scopes=vgg_16/fc8 \
#    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
#    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
#    --train_image_size=299 \



python3 slim/train_by_tfrecord_BN.py \
    --train_dir=TSN_F_all1 \
    --dataset_dir=/data2/jyli/UCF101/set1/train/tfrecord \
    --model_name=inception_v3_frozen_BN \
    --checkpoint_path=TSN_F_lastlayer1 \
    --max_number_of_steps=4500 \
    --epoch_size=9537  \
    --num_classes=101  \
    --batch_size=32 \
    --num_clones=4  \
    --agg_fn=average \
    --topK=3  \
    --num_readers=4  \
    --segment_num=5  \
    --train_image_size=240 \
    --learning_rate=0.001 \
    --num_epochs_per_decay=5 \
    --learning_rate_decay_factor=0.15 \
    --learning_rate_decay_type=exponential \
    --dropout_keep=0.5  \
    --save_interval_secs=300 \
    --save_summaries_secs=300 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.0005  