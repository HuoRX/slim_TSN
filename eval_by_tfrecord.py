# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory, dataset_factory_by_me, eval_score
from mynet import nets_factory
import preprocessing
import itertools
import numpy as np
import threading
import os


slim = tf.contrib.slim


tf.app.flags.DEFINE_integer(
        'batch_size', 8, 'The number of samples in each batch.')


tf.app.flags.DEFINE_integer(
        'num_classes', 101,
        'The number of class.')


tf.app.flags.DEFINE_string(
        'dataset_dir', None, 'The tfrecord file where the dataset are stored.')


tf.app.flags.DEFINE_integer(
        'segment_num', 10,
        'The number of segments in one video.')


tf.app.flags.DEFINE_integer(
        'max_num_batches', None,
        'Max number of batches to evaluate by default use all.')


tf.app.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')


tf.app.flags.DEFINE_string(
        'checkpoint_path', '/tmp/tfmodel/',
        'The directory where the model was written to or an absolute path to a '
        'checkpoint file.')


tf.app.flags.DEFINE_string(
        'eval_dir', '/tmp/model/', 'Directory where the results are saved to.')



tf.app.flags.DEFINE_integer(
        'num_preprocessing_threads', 4,
        'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
        'num_readers', 2,
        'The number of threads used to create the batches.')


tf.app.flags.DEFINE_integer(
        'labels_offset', 0,
        'An offset for the labels in the dataset. This flag is primarily used to '
        'evaluate the VGG and ResNet architectures which do not use a background '
        'class for the ImageNet dataset.')


tf.app.flags.DEFINE_string(
        'model_name', 'inception_v3', 'The name of the architecture to evaluate.')


tf.app.flags.DEFINE_string(
        'preprocessing_name', None, 'The name of the preprocessing to use. If left '
        'as `None`, then the model_name flag is used.')


tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')


tf.app.flags.DEFINE_integer(
        'eval_image_size', 299, 'Eval image size')


tf.app.flags.DEFINE_integer(
        'read_H', 240, "The height of raw input image.")


tf.app.flags.DEFINE_integer(
        'read_W', 320, "The width of raw input image.")

tf.app.flags.DEFINE_integer(
        'recall_at',5, "The size of image when saved in the tfrecord, please check 'tfrecord_info.txt',"
                                   "created when generated the tfrecord.")

tf.app.flags.DEFINE_integer(
        'sample_size',1000, "The number of samples used to test.")


tf.app.flags.DEFINE_string(
        'agg_fn', 'average',
        'The aggregation function used to predict from segments,'
        'one of "average","max","topKpool".'
        'If use "topKpool", please specify K by --topK=')

tf.app.flags.DEFINE_integer(
        'topK', 5,
        'The top K logits .')


FLAGS = tf.app.flags.FLAGS


def ID_dic(ID, label_text=None):
    if label_text is None:
        label_text = '/home/herbert/python_project/TSN/data/ucf101/split2/train/category.txt'

    label_list = open(label_text)
    lines = label_list.readlines()
    lines = lines[1:-1]
    # print(len(lines))
    # print(lines[-1])
    id = lines[ID]
    name = id.split(',')[0]
    return name


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        #########################
        # Initialize parameters #
        #########################

        dataset_dir = FLAGS.dataset_dir
        segment_num = FLAGS.segment_num
        batch_size = FLAGS.batch_size
        sample_size = FLAGS.sample_size
        num_classes = FLAGS.num_classes
        agg_fn = eval_score.get_agg_fn(FLAGS.agg_fn)

        if FLAGS.recall_at > num_classes:
            recall_at = num_classes
        else:
            recall_at = FLAGS.recall_at

        if FLAGS.eval_dir == None:
            eval_dir = os.path.join(os.path.dirname(FLAGS.checkpoint_path),'eval_dir')
            os.mkdir(eval_dir)
        else:
            eval_dir = FLAGS.eval_dir

        frame_name_list = []
        for i in range(segment_num):
            frame_name_list.append('img'+str(i))

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                                num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                                                is_training=False)


        #####################################
        # Select the preprocessing function #
        #####################################
        # preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        # image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        #         preprocessing_name,
        #         is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        # read_image_size = FLAGS.readin_image_size or eval_image_size

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        frames_batch, labels, height_read, width_read, id = preprocessing.train_inputs(
                                                                dataset_dir, segment_num=segment_num,
                                                                batch_size=batch_size,
                                                                num_preprocess_threads=FLAGS.num_preprocessing_threads,
                                                                train=False, train_image_size=eval_image_size)

        # img_ori_batch = tf.cast(img_ori_batch, dtype=tf.uint8)
        # if img_ori_batch.dtype != tf.float32:
        #    img_ori_batch = tf.image.convert_image_dtype(img_ori_batch, dtype=tf.float32)


        # preprocessing
        #processed_images = tf.subtract(img_ori_batch, 0.5)
        #processed_images = tf.multiply(processed_images, 2.0)

        # processed_images = img_ori_batch

        #frame_batch = list()
        #
        # crop 4 corners and center (with flip flop, total input shape is [10*segment, img_h, img_w, 3] for one video)
        # crop_size = [segment_num, eval_image_size, eval_image_size, 3]
        # crop_start = [[0, 0, 0, 0],                                                                # left top corner
        #              [0, height_read - eval_image_size, 0, 0],                                # left bottom corner
        #              [0, 0, width_read- eval_image_size, 0],                                  # right top corner
        #              [0, height_read - eval_image_size, width_read - eval_image_size, 0],     # right bottom corner
        #              ]        # center

        # for i in range(batch_size):
        #     processing_images = processed_images[i]

        #     cropped_batch = tf.slice(processing_images, begin=crop_start[0], size=crop_size)
        #     for ii in range(1, 5):
        #       cropped_img = tf.slice(processing_images, begin=crop_start[ii], size=crop_size)
        #       cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        #     flipped_img = tf.reshape(processing_images, [segment_num*height_read, width_read, 3])
        #     flipped_img = tf.image.flip_left_right(flipped_img)
        #     flipped_img = tf.reshape(flipped_img,[segment_num, height_read, width_read, 3])

        #     for ii in range(5):
        #       cropped_img = tf.slice(flipped_img, begin=crop_start[ii], size=crop_size)
        #       cropped_batch = tf.concat([cropped_batch, cropped_img], axis=0)

        #     processed_batch = tf.reshape(cropped_batch, [10*segment_num, eval_image_size, eval_image_size, 3])
        #     processed_batch = tf.expand_dims(processed_batch, axis=0)

        #     frame_batch.append(processed_batch)


        ####################
        # Define the model #
        ####################

        frames_group = tf.reshape(frames_batch, [batch_size * segment_num * 10, eval_image_size, eval_image_size, 3])



        logit_out, _ = network_fn(frames_group)

        logit_out = tf.reshape(logit_out, [batch_size, segment_num*10, num_classes])
        logit_agg = agg_fn(logit_out, k=FLAGS.topK)

        logit_list = tf.reshape(logit_agg, [batch_size, num_classes])


        predictions = tf.argmax(logit_list, 1)
        predictions = tf.squeeze(tf.reshape(predictions, [batch_size,-1]))
        labels = tf.reshape(labels, [-1])

        correct = tf.reduce_sum(tf.cast(tf.equal(predictions,labels), tf.int32))


        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                    FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                    slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


        with tf.Session() as sess:
            import time
            time1 = time.time()
            sess.run(tf.global_variables_initializer())

            init_fn(sess)

            if FLAGS.max_num_batches:
                num_batches = FLAGS.max_num_batches
            else:
                # This ensures that we make a single pass over all of the data.
                num_batches = math.ceil(sample_size / float(batch_size))
                print("Number of samples:", sample_size)
                print("Number of batches:", num_batches)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # label_l = list()
            # id_list = list()
            #
            # for i in range(num_batches*2):
            #     a,b = sess.run([logit_list, id])
            #     print(a)
            #     print(b)
            #     # id_list.append(b)
            #
            # print(id_list)

            ac_correct = 0
            for i in range(num_batches):
                print('[{}/{}]'.format(i+1, num_batches))
                a = sess.run(correct)
                ac_correct += a

            accuracy = ac_correct/sample_size

            out = 'Accuracy:{:.3f}'.format(accuracy)
            print(out)
            save_path = FLAGS.checkpoint_path

            open(save_path + '/' + 'Accuracy_{}_{}_bytfrecord.txt'.format(FLAGS.model_name, FLAGS.agg_fn), 'w').writelines(out)

            time2 = time.time()
            print('Time cost:', time2 - time1)

            coord.request_stop()
            coord.join(threads)





if __name__ == '__main__':
    tf.app.run()

    # # # #
    # aa = tf.reshape(frames_group, [batch_size, segment_num*10, eval_image_size, eval_image_size, 3])
    # # bb  = tf.reshape(b, [batch_size, segment_num, num_classes])
    # # bb = eval_score.average_pool3D(bb)

    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #
    #     for i in range(1):
    #         m = sess.run(frames_group)
    #         print(m.shape)
    #         # m1 = sess.run(a,)
    #         # print('in1',np.argmax(n, axis=1))
    #         # print('in2',np.argmax(n1, axis=1))
    #     # print(m1.shape)
    #
    #     m,n = sess.run([aa,labels])
    #     # n = sess.run(labels)
    #     # print(n)
    #     print(m.shape)
    #     x = m
    #     print(n)
    #     # a_val, b_val= sess.run([a_batch, b_batch,])
    #     # print('second batch:')
    #     # print('  a_val:',a_val.shape)
    #     # print('  b_val:',b_val)
    #
    #     coord.request_stop()
    #     coord.join(threads)

    # import matplotlib.pylab as plt
    #
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(x[0][0])
    # plt.subplot(222)
    # plt.imshow(x[0][120])
    # plt.subplot(223)
    # plt.imshow(x[0][180])
    # plt.subplot(224)
    # plt.imshow(x[0][240])
    #
    # # print(np.argmax(n, axis=1))
    #
    # plt.figure(2)
    # plt.subplot(221)
    # plt.imshow(x[1][0])
    # plt.subplot(222)
    # plt.imshow(x[1][12])
    # plt.subplot(223)
    # plt.imshow(x[1][18])
    # plt.subplot(224)
    # plt.imshow(x[1][240])
    #
    #
    # # plt.figure(3)
    # # plt.imshow(x[2][0])
    # # # print(np.argmax(n, axis=1))
    # #
    # # plt.figure(4)
    # # plt.imshow(x[3][0])
    #
    # plt.show()