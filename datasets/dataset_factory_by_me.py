# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from random import shuffle
import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_integer('batch_size', 8,
#                             """Number of videos to process in a batch.""")
# tf.app.flags.DEFINE_integer('frame_size', 224,
#                             """Provide square images of this size.""")
# tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
#                             """Number of preprocessing threads per tower. """
#                             """Please make this a multiple of 4.""")
#
# tf.app.flags.DEFINE_integer('num_readers', 4,
#                             """Number of parallel readers during train.""")

# tf.app.flags.DEFINE_integer('num_segments', 3,
#                             'num_segments for tsn')

tf.app.flags.DEFINE_integer('num_length', 1,
                            'num_length for tsn')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def get_datasetfromlist(train_list_file, shuffle_=False):
    # in train_list_file, the order of each line should be: [frame path, frame number, label]
    f = open(train_list_file)
    lines = f.readlines()
    # if shuffle_ == True:
    #     shuffle(lines)
    video_list = []
    labels = []
    for line in lines:
        video_path = line.split('.')[0]
        label = line.split(',')[-1]
        label = label.replace('\n','')
        # print(label)
        # print(len(label))
        frame_path = video_path.replace('video', 'frame')
        frames = sorted(os.listdir(frame_path))
        # print(frames)
        frame_list = list(map(lambda x:frame_path+'/'+x,frames))
        # print(frame_list)
        video_list.append(frame_list)
        labels.append(int(label))
    return video_list, labels


def convert_trainlist2datapath_test(path_txt):
    path = path_txt
    f = open(path)
    lines = f.readlines()
    video_num = len(lines)
    # print('Total video number:', video_num)
    a = []
    b = []
    segment_num = len(lines[0].split(','))-1
    for i in range(video_num):
        p_r = lines[i].split(',')
        l_r = int(p_r[-1])
        b.append(l_r)
        a.append(p_r[:-1])
        # print(a)
        # print(b)

    return a,b,segment_num


def convert_trainlist2datapath(path_txt):
    path_in = path_txt
    f = open(path_in)
    lines = f.readlines()
    shuffle(lines)
    check = lines[0].find(',')

    if check != -1:
        print('Load train list:', path_in)
        video_num = len(lines)
        # print('Total video number:', video_num)
        a = []
        b = []
        segment_num = len(lines[0].split(','))-1
        for i in range(video_num):
            p_r = lines[i].split(',')
            l_r = int(p_r[-1])
            b.append(l_r)
            a.append(p_r[:-1])

        return a,b,segment_num


    else:
        line_total = []
        a_total = []
        b_total = []
        for path in lines:
            path = path.replace('\n', '')
            print('Load train list:', path)
            ff = open(path)
            lines_ = ff.readlines()
            line_total += lines_
            segment_num = len(lines_[0].split(',')) - 1

        for i in range(7):
            shuffle(line_total)

        for line in line_total:
            p_r = line.split(',')
            l_r = int(p_r[-1])
            b_total.append(l_r)
            a_total.append(p_r[:-1])

        return a_total, b_total, segment_num

    # else:
    #     a_total = []
    #     b_total = []
    #     shuffle(lines)
    #     for path in lines:
    #         path = path.replace('\n','')
    #         print('Load train list:', path)
    #         a = []
    #         b = []
    #         ff = open(path)
    #         lines_ = ff.readlines()
    #         video_num = len(lines_)
    #         segment_num = len(lines_[0].split(',')) - 1
    #         for i in range(video_num):
    #             p_r = lines_[i].split(',')
    #             l_r = int(p_r[-1])
    #             b.append(l_r)
    #             a.append(p_r[:-1])
    #
    #         a_total += a
    #         b_total += b
    #
    #     return a_total, b_total, segment_num



# p = '/home/herbert/python_project/TSN/data/dataset1/train/train_list.txt'
#
# a,b,c = convert_trainlist2datapath(p)
# print(c)
# print(a[:5])
# print(b[:5])


def train_inputs(dataset, batch_size=None, num_preprocess_threads=None, segment_num=None, train=True):
    """Generate batches of videos  for training.

    Use this function as the inputs for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      dataset: instance of Dataset class specifying the dataset.
      batch_size: integer, number of examples in batch
      num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.

    Returns:
      videos: 4D tensor of size [batch_size*num_frames, FLAGS.frame_size,
                                         frame_size, 3].
      labels: 1-D integer Tensor of [FLAGS.batch_size,num_class].
    """
    if not batch_size:
        batch_size = FLAGS.batch_size

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        videos, labels, height, width, id = batch_inputs(
            dataset, batch_size, train=train, segment_num=segment_num,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=FLAGS.num_readers
            )
    return videos, labels, height, width, id


def SampleTsnFrames(video_buffer, num_segments, num_length, height_ori, width_ori):
    height = height_ori
    width = width_ori

    num_frames = tf.shape(video_buffer)[0]
    frame_index_offset = tf.range(num_length)
    frame_index_offset = tf.squeeze(frame_index_offset)
    max_start_frame_index = tf.maximum(num_frames // num_segments - num_length, 0)
    frame_index = None
    for i in range(num_segments):
        segment_start_frame_index = tf.cast(tf.multiply(tf.random_uniform([1]),
                                                        tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
        frame_index_temp = tf.minimum(
            segment_start_frame_index + frame_index_offset + tf.cast(i * (num_frames // num_segments), tf.int32),
            tf.cast(num_frames - 1, tf.int32))
        if i == 0:
            frame_index = frame_index_temp
        else:
            frame_index = tf.concat([frame_index, frame_index_temp], 0)
    images = None
    for i in range(num_length * num_segments):
        image = tf.image.decode_jpeg(video_buffer[frame_index[i]], channels=3)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [height, width])
        # image = tf.cast(image, dtype=tf.uint8)
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)

        if i == 0:
            images = tf.expand_dims(image, 0)
        else:
            images = tf.concat([images, tf.expand_dims(image, 0)], 0)

    return images


def SampleTsnFrames_eval(video_buffer, num_segments, num_length, height_ori, width_ori):
    height = height_ori
    width = width_ori

    num_frames = tf.shape(video_buffer)[0]
    frame_index_offset = tf.range(num_length)
    frame_index_offset = tf.squeeze(frame_index_offset)
    # max_start_frame_index = tf.maximum(num_frames // num_segments - num_length, 0)
    frame_index = None
    for i in range(num_segments):
        frame_index_temp = tf.minimum(frame_index_offset + tf.cast(i * (num_frames // (num_segments-1)), tf.int32),
            tf.cast(num_frames - 1, tf.int32))
        if i == 0:
            frame_index = frame_index_temp
        else:
            frame_index = tf.concat([frame_index, frame_index_temp], 0)
    images = None
    for i in range(num_length * num_segments):
        image = tf.image.decode_jpeg(video_buffer[frame_index[i]], channels=3)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [height, width])
        # image = tf.cast(image, dtype=tf.uint8)
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)

        if i == 0:
            images = tf.expand_dims(image, 0)
        else:
            images = tf.concat([images, tf.expand_dims(image, 0)], 0)

    return images


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      height: 462
      width: 581
      channels: 3
      labelS: 615
      video_id: 'n03623198'
      data: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      video_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: sparse Tensor containing the label.
    """
    # Dense features in Example proto.
    contexts, features = tf.parse_single_sequence_example(
        example_serialized,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
            "width": tf.FixedLenFeature([],tf.int64),
            "height": tf.FixedLenFeature([],tf.int64),
            "channels": tf.VarLenFeature(tf.int64),
            "labels": tf.FixedLenFeature([],tf.int64)},
        sequence_features={
            'data': tf.FixedLenSequenceFeature([], dtype=tf.string)
        })
    video_id = contexts["video_id"]
    width = tf.cast(contexts["width"], dtype=tf.int32)
    height = tf.cast(contexts["height"], dtype=tf.int32)
    labels = contexts["labels"]
    video_buffer = features["data"]

    return video_buffer, labels, height, width, video_id


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None, segment_num=None,
                 num_readers=1):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
      dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers

    Returns:
      videos: 4D tensor of size [batch_size*num_frames, FLAGS.frame_size,
                                         frame_size, 3].
      labels: 1-D integer Tensor of [FLAGS.batch_size,num_class].
    Raises:
      ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = tf.gfile.Glob(os.path.join(dataset, '*'))
        data_files = [os.path.join(dataset, file) for file in data_files]

        data_files = data_files
        num_classes = 101
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=True,
                                                            capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)
        if num_preprocess_threads is None:
            num_preprocess_threads = FLAGS.num_preprocess_threads

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = FLAGS.num_readers

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 3

        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)


        videos_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the videos and metadata.
            video_buffer, label_index, height, width, ID = parse_example_proto(
                example_serialized)

            height = FLAGS.read_H
            width = FLAGS.read_W

            if train:
                frames = SampleTsnFrames(video_buffer, segment_num, FLAGS.num_length, height, width)
            else:
                frames = SampleTsnFrames_eval(video_buffer, segment_num, FLAGS.num_length, height, width)

            frames = tf.reshape(frames, [segment_num, height, width, 3])

            # dense_label = tf.cast(tf.sparse_to_indicator(label_index, num_classes), tf.float32)
            # dense_label.set_shape([num_classes])
            dense_label = label_index
            dense_label = tf.reshape(dense_label, [1])

            videos_and_labels.append([frames, dense_label, ID])

        videos, dense_labels, id_out = tf.train.batch_join(
            videos_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape frames into these desired dimensions.
        # height = FLAGS.frame_size
        # width = FLAGS.frame_size
        height = FLAGS.read_H
        width = FLAGS.read_W
        depth = 3

        videos = tf.cast(videos, tf.float32)
        videos = tf.reshape(videos, shape=[-1, height, width, depth])

        return videos, tf.reshape(dense_labels, [batch_size]), height, width, id_out