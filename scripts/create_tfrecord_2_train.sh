#!/usr/bin/env bash

python3 tools/build_video_data.py \
  --train_directory='/data2/jyli/UCF101/set2/train/video' \
  --output_directory='set2/train/tfrecord' \
  --train_shards=10 \
  --file_type=video  \
  --extract=rgb  \
  --fps=1  \
  --labels_file='/data2/jyli/UCF101/set2/train/label.txt'