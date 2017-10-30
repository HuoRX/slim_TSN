#!/usr/bin/env bash

python3 tools/build_video_data.py \
  --train_directory='/data2/jyli/UCF101/set3/test/video' \
  --output_directory='set3/test/tfrecord' \
  --train_shards=10 \
  --file_type=video  \
  --extract=rgb  \
  --fps=1  \
  --labels_file='/data2/jyli/UCF101/set3/test/label.txt'