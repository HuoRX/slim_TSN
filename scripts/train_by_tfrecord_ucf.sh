#!/bin/bash

# The commands below fine tune the pretrained VGG16_slim 

# DATASET_DIR = train data location
# TRAIN_DIR = where the new checkpoint save
# CHECKPOINT_PATH = net/VGG_slim/vgg_16.ckpt  checkpoint file for pretrain net
#    --checkpoint_exclude_scopes=vgg_16/fc8 \

python3 slim/train_by_tfrecord.py \
    --train_dir=/home/jingyu/Herbert/python_project/TSN/data/dataset1/ckpt_slim/bytfrecord1 \
    --dataset_dir=/home/jingyu/Herbert/python_project/TSN/data/dataset1/train/tfrecord \
    --model_name=inception_v3 \
    --checkpoint_path=build_net/nets/checkpoint/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps=1500 \
    --epoch_size=203  \
    --num_classes=2  \
    --batch_size=16 \
    --agg_fn=average \
    --topK=3  \
    --num_readers=4  \
    --segment_num=5  \
    --train_image_size=224 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=32 \
    --save_summaries_secs=32 \
    --log_every_n_steps=32 \
    --optimizer=adam \
    --weight_decay=0.00004
