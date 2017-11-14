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
from mynet import nets_factory

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('1', 1, '')

#tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
#                           """Size of the queue of preprocessed images. """
#                            """Default is ideal but try smaller values, e.g. """
#                            """4, 2 or 1, if host memory is constrained. See """
#                            """comments in code for more details.""")

tf.app.flags.DEFINE_string('input_type', 'RGB',
                           'The type of input: RGB, RGBDiff, OpticalFlow')


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
    if FLAGS.input_type == "OpticalFlow":
        contexts, features = tf.parse_single_sequence_example(
            example_serialized,
            context_features={"video_id": tf.FixedLenFeature(
                [], tf.string),
                "width": tf.FixedLenFeature([], tf.int64),
                "height": tf.FixedLenFeature([], tf.int64),
                "channels": tf.VarLenFeature(tf.int64),
                "labels": tf.FixedLenFeature([], tf.int64)},
            sequence_features={
                'data': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'flow_x': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'flow_y': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })
        flow_x_buffer = features["flow_x"]
        flow_y_buffer = features["flow_y"]
    else:
        contexts, features = tf.parse_single_sequence_example(
            example_serialized,
            context_features={"video_id": tf.FixedLenFeature(
                [], tf.string),
                "width": tf.FixedLenFeature([], tf.int64),
                "height": tf.FixedLenFeature([], tf.int64),
                "channels": tf.VarLenFeature(tf.int64),
                "labels": tf.FixedLenFeature([], tf.int64)},
            sequence_features={
                'data': tf.FixedLenSequenceFeature([], dtype=tf.string),

            })
        flow_x_buffer = None
        flow_y_buffer = None
    video_id = contexts["video_id"]
    width = tf.cast(contexts["width"], dtype=tf.int32)
    height = tf.cast(contexts["height"], dtype=tf.int32)
    labels = contexts["labels"]
    frame_buffer = features["data"]
    video_buffer = [frame_buffer, [flow_x_buffer, flow_y_buffer]]

    return video_buffer, labels, height, width, video_id


def OpticalFlow(video_buffer, frame_index, segment_number):
    flows = None
    for i in range(segment_number):
        flow_xy = None
        index = frame_index[i] - tf.cast((1 - 0.5) // 2, tf.int32)
        for ii in range(1):
            flow_x = tf.image.decode_jpeg(video_buffer[0][index + ii], channels=1)
            flow_x = tf.image.convert_image_dtype(flow_x, dtype=tf.float32)
            flow_x = tf.image.resize_images(flow_x, size=[240, 320])

            flow_y = tf.image.decode_jpeg(video_buffer[1][index + ii], channels=1)
            flow_y = tf.image.convert_image_dtype(flow_y, dtype=tf.float32)
            flow_y = tf.image.resize_images(flow_y, size=[240, 320])
            # print('flow_y', flow_y)

            flow_xy_temp = tf.concat([flow_x, flow_y], axis=-1)
            flow_xy_temp = tf.reshape(flow_xy_temp, [240, 320, 2])

            if ii == 0:
                flow_xy = flow_xy_temp
            else:
                flow_xy = tf.concat([flow_xy, flow_xy_temp], axis=-1)

        flow_xy = tf.reshape(flow_xy, [240, 320, 2 * 1])

        if i == 0:
            flows = tf.expand_dims(flow_xy, 0)
        else:
            flows = tf.concat([flows, tf.expand_dims(flow_xy, 0)], 0)

    return flows


def RGBDiff(video_buffer, frame_index, segment_number):
    raw_frames = None
    raw_frames1 = None
    raw_frames2 = None
    for i in range(segment_number):
        frame1 = tf.image.decode_jpeg(video_buffer[frame_index[i]], channels=3, dct_method='INTEGER_FAST')
        frame1 = tf.image.convert_image_dtype(frame1, dtype=tf.float32)
        frame1 = tf.image.resize_images(frame1, size=[240, 320])

        frame2 = tf.image.decode_jpeg(video_buffer[frame_index[i] + 1], channels=3, dct_method='INTEGER_FAST')
        frame2 = tf.image.convert_image_dtype(frame2, dtype=tf.float32)
        frame2 = tf.image.resize_images(frame2, size=[240, 320])

        frame_diff = tf.subtract(frame1, frame2)

        if not FLAGS.two_stream:
            if i == 0:
                raw_frames = tf.expand_dims(frame_diff, 0)
            else:
                raw_frames = tf.concat([raw_frames, tf.expand_dims(frame_diff, 0)], 0)

        else:
            if i == 0:
                raw_frames1 = tf.expand_dims(frame1, 0)
                raw_frames2 = tf.expand_dims(frame_diff, 0)
            else:
                raw_frames1 = tf.concat([raw_frames1, tf.expand_dims(frame1, 0)], 0)
                raw_frames2 = tf.concat([raw_frames2, tf.expand_dims(frame_diff, 0)], 0)

    if FLAGS.two_stream:
        raw_frames = [raw_frames1, raw_frames2]

    return raw_frames


def RGB(video_buffer, frame_index, segment_number):
    raw_frames = None
    for i in range(segment_number):
        frame = tf.image.decode_jpeg(video_buffer[frame_index[i]], channels=3, dct_method='INTEGER_FAST')
        frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
        frame = tf.image.resize_images(frame, size=[240, 320])

        if i == 0:
            raw_frames = tf.expand_dims(frame, 0)
        else:
            raw_frames = tf.concat([raw_frames, tf.expand_dims(frame, 0)], 0)

    return raw_frames


def train_inputs(dataset, batch_size=None, num_preprocess_threads=None, segment_num=None, train=True, train_image_size=None):
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
            num_readers=FLAGS.num_readers,
            train_image_size=train_image_size
        )
    return videos, labels, height, width, id


def SampleTsnFrames(video_buffer, num_segments, height_ori, width_ori):
    height = height_ori
    width = width_ori

    num_frames = tf.shape(video_buffer[0])[0]
    # print('num_frame', num_frames)
    start_offset = tf.cast((1 - 0.5) // 2, tf.int32)
    end_offset = tf.cast(1 // 2, tf.int32)
    frame_index = None
    step = tf.maximum((num_frames) // (num_segments), 0)
    for i in range(num_segments):
        segment_start_frame_index = tf.cast(tf.multiply(tf.random_uniform([1]),
                                                        tf.cast(step - 1 + 1, tf.float32)), tf.int32)
        frame_index_temp = tf.minimum(start_offset + segment_start_frame_index + tf.cast(i * step, tf.int32),
                                      tf.cast((i + 1) * step - end_offset, tf.int32))

        if i == 0:
            frame_index = frame_index_temp
        else:
            frame_index = tf.concat([frame_index, frame_index_temp], 0)
    frame_index = tf.reshape(frame_index, [num_segments])

    if FLAGS.input_type == 'OpticalFlow':
        video_info = video_buffer[1]
        raw_frames = OpticalFlow(video_buffer=video_info, frame_index=frame_index,
                                 segment_number=num_segments)
    else:
        video_info = video_buffer[0]
        if FLAGS.input_type == 'RGBDiff':
            raw_frames = RGBDiff(video_buffer=video_info, frame_index=frame_index, segment_number=num_segments)
        else:
            raw_frames = RGB(video_buffer=video_info, frame_index=frame_index, segment_number=num_segments)

    # if FLAGS.input_type == 'OpticalFlow':
    #     channel = 2 * 1
    # else:
    #     channel = 3

    return raw_frames


def SampleTsnFrames_eval(video_buffer, num_segments, height_ori, width_ori):
    height = 240
    width = 320
    num_frames = num_segments

    frame_index = None
    start_offset = tf.cast((1 - 0.5) // 2, tf.int32)
    end_offset = tf.cast(1 // 2, tf.int32)
    total_frames = tf.shape(video_buffer[0])[0]

    step = tf.maximum((total_frames - 1) // (num_frames - 1), 0)
    for i in range(num_frames):
        frame_index_temp = tf.minimum(tf.cast(i * step + start_offset, tf.int32),
                                      tf.cast(total_frames - end_offset, tf.int32))
        frame_index_temp = tf.expand_dims(frame_index_temp, axis=0)

        if i == 0:
            frame_index = frame_index_temp
        else:
            frame_index = tf.concat([frame_index, frame_index_temp], 0)
    frame_index = tf.reshape(frame_index, [num_frames])

    if FLAGS.input_type == 'OpticalFlow':
        channel = 2 * 1
        video_info = video_buffer[1]
        raw_frames = OpticalFlow(video_buffer=video_info, frame_index=frame_index,
                                 segment_number=num_frames)

    else:
        channel = 3
        video_info = video_buffer[0]
        if FLAGS.input_type == 'RGBDiff':
            raw_frames = RGBDiff(video_buffer=video_info, frame_index=frame_index, segment_number=num_frames)
        else:
            raw_frames = RGB(video_buffer=video_info, frame_index=frame_index, segment_number=num_frames)

    raw_frames = tf.reshape(raw_frames, [num_frames, 240, 320, channel])

    return raw_frames

def preprocessing_for_train(frames, segment_num, channel, train_image_size):
    ##################
    # preprocesssing #
    ##################

    # img_ori_batch = tf.cast(img_ori_batch, dtype=tf.uint8)
    # print('loading image size:', img_ori_batch)
    # if img_ori_batch.dtype != tf.float32:
    #    img_ori_batch = tf.image.convert_image_dtype(img_ori_batch, dtype=tf.float32)


    """

    :type videos_and_labels: object
    """
    height = FLAGS.read_H
    width = FLAGS.read_W

    frames = tf.reshape(frames,[segment_num, height, width, channel])
    frames = tf.image.resize_images(frames, [256, 340])

    # preprocessing
    # crop the image into network input size
    random_l = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None)
    random_l = random_l[0]
    random_s = tf.random_uniform([1], minval=0, maxval=5, dtype=tf.int32, seed=None, name=None)
    random_s = random_s[0]
    crop_length = tf.convert_to_tensor([240, 224, 192, 168], dtype=tf.int32)
    crop_l = crop_length[random_l]
    crop_size = [segment_num, crop_l, crop_l, channel]
    center_h = tf.div(tf.subtract(256, crop_l), 2)
    center_w = tf.div(tf.subtract(340, crop_l), 2)

    crop_start_all = tf.convert_to_tensor([[0, 0, 0, 0],
                                           [0, 0, 340 - crop_l, 0],
                                           [0, 256 - crop_l, 0, 0],
                                           [0, 256 - crop_l, 340 - crop_l, 0],
                                           [0, center_h, center_w, 0]])

    crop_start = crop_start_all[random_s]

    cropped_frames = tf.slice(frames, begin=crop_start, size=crop_size)

    cropped_frames = tf.image.resize_images(cropped_frames, [train_image_size, train_image_size])

    # raondomly flip left and right
    flip_frames = tf.reshape(cropped_frames,
                             [segment_num * train_image_size, train_image_size, channel])
    flip_frames = tf.image.random_flip_left_right(flip_frames)
    flip_frames = tf.reshape(flip_frames,
                             [segment_num, train_image_size, train_image_size, channel])

    processed_frames = tf.subtract(flip_frames, 0.5)
    processed_frames = tf.multiply(processed_frames, 2.0)

    # dense_label = tf.cast(tf.sparse_to_indicator(label_index, num_classes), tf.float32)
    # dense_label.set_shape([num_classes])

    return processed_frames



def preprocessing_for_eval(frames, segment_num, channel, eval_image_size):

    # frames = tf.image.resize_images(frames, [256, 340])
    # preprocessing
    # crop the image into network input size

    height = FLAGS.read_H
    width = FLAGS.read_W

    frames = tf.reshape(frames, [segment_num, height, width, channel])

    processed_frames = tf.subtract(frames, 0.5)
    processed_frames = tf.multiply(processed_frames, 2.0)

    frame_batch = list()

    crop_size = [segment_num, eval_image_size, eval_image_size, channel]

    crop_start = [[0, 0, 0, 0],  # left top corner
                  [0, height - eval_image_size, 0, 0],  # left bottom corner
                  [0, 0, width - eval_image_size, 0],  # right top corner
                  [0, height - eval_image_size, width - eval_image_size, 0],  # right bottom corner
                  [0, tf.cast((height - eval_image_size) / 2, tf.int32),
                   tf.cast((width - eval_image_size) / 2, tf.int32), 0]]  # center

    for i in range(1):
        processing_images = processed_frames

        cropped_batch = tf.slice(processing_images, begin=crop_start[0], size=crop_size)
        for ii in range(1, 5):
            cropped_img = tf.slice(processing_images, begin=crop_start[ii], size=crop_size)
            cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        flipped_img = tf.reshape(processing_images, [segment_num * height, width, channel])
        flipped_img = tf.image.flip_left_right(flipped_img)
        flipped_img = tf.reshape(flipped_img, [segment_num, height, width, channel])

        for ii in range(5):
            cropped_img = tf.slice(flipped_img, begin=crop_start[ii], size=crop_size)
            cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        processed_batch = tf.reshape(cropped_batch, [10 * segment_num, eval_image_size, eval_image_size, channel])
        processed_batch = tf.expand_dims(processed_batch, axis=0)

        frame_batch.append(processed_batch)


    return frame_batch



def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None, segment_num=None,
                 num_readers=1, train_image_size=None):
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
        if FLAGS.input_type == 'OpticalFlow':
            channel = 2 * 1
        else:
            channel = 3
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

        min_queue_examples = examples_per_shard * 16
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
                frames = SampleTsnFrames(video_buffer, segment_num, height, width)
                frames = preprocessing_for_train(frames, segment_num, channel, train_image_size)
                frames = tf.reshape(frames, [segment_num, train_image_size, train_image_size, channel])

            else:
                frames = SampleTsnFrames_eval(video_buffer, segment_num, height, width)
                frames = preprocessing_for_eval(frames, segment_num, channel, train_image_size)
                frames = tf.reshape(frames, [10*segment_num, train_image_size, train_image_size, channel])



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

        videos = tf.cast(videos, tf.float32)
        if train:
            videos = tf.reshape(videos, shape=[batch_size*segment_num, train_image_size, train_image_size, channel])

        else:
            videos = tf.reshape(videos, shape=[batch_size * segment_num*10, train_image_size, train_image_size, channel])

        return videos, tf.reshape(dense_labels, [batch_size]), train_image_size, train_image_size, id_out
