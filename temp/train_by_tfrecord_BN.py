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
"""Generic training script that trains a model using a given dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import preprocessing
from deployment import model_deploy
from mynet import nets_factory

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
        'train_dir', '/tmp/tfmodel/',
        'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
        'num_clones', 1, 'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean(
        'clone_on_cpu', False, 'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer(
        'worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
        'num_ps_tasks', 0,
        'The number of parameter servers. If the value is 0, then the parameters '
        'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
        'num_readers',4, 'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
        'num_preprocessing_threads', 4,
        'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
        'log_every_n_steps', 10,
        'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
        'save_summaries_secs', 600,
        'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
        'save_interval_secs', 600,
        'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
        'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
        'weight_decay', 0.0005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float(
        'dropout_keep', 0.8, 'The dropout keep ratio.')

tf.app.flags.DEFINE_string(
        'optimizer', 'momentum',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
        'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float(
        'ftrl_learning_rate_power', -0.5, 'The learning rate power.')


tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float(
        'rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float(
        'rmsprop_decay', 0.9, 'Decay term for RMSProp.')


#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

tf.app.flags.DEFINE_float(
        'learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
        'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
        'num_steps_per_decay', None,
        'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
        'moving_average_decay', 0.9,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')

##################
# Dataset Flags  #
##################

tf.app.flags.DEFINE_string(
        'dataset_dir', None, 'The directory where the tfrecord are stored.')

tf.app.flags.DEFINE_integer(
        'num_classes', 101,
        'The number of class.')

tf.app.flags.DEFINE_integer(
        'segment_num', 10,
        'The number of segments in one video.')

tf.app.flags.DEFINE_integer(
        'labels_offset', 0,
        'An offset for the labels in the dataset. This flag is primarily used to '
        'evaluate the VGG and ResNet architectures which do not use a background '
        'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
        'model_name', 'inception_v3_frozen_BN', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
        'preprocessing_name', None, 'The name of the preprocessing to use. If left '
        'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
        'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
        'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer(
        'read_H', 240, "The height of raw input image.")


tf.app.flags.DEFINE_integer(
        'read_W', 320, "The width of raw input image.")

tf.app.flags.DEFINE_integer(
        'max_number_of_steps', 1000, 'The maximum number of training steps.')

tf.app.flags.DEFINE_integer(
        'max_number_of_epochs', None,
        'The maximum number of training epoch.')

tf.app.flags.DEFINE_integer(
        'epoch_size', 1000,
        'The number of sample in an epoch, can choose it equal to the total sample size.')


#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
        'agg_fn', 'average',
        'The aggregation function used to predict from segments,'
        'one of "average","max","topKpool".'
        'If use "topKpool", please specify K by --topK=')

tf.app.flags.DEFINE_integer(
        'topK', 5,
        'The top K logits .')



tf.app.flags.DEFINE_string(
        'checkpoint_path', None,
        'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
        'checkpoint_exclude_scopes', None,
        'Comma-separated list of scopes of variables to exclude when restoring '
        'from a checkpoint.')

tf.app.flags.DEFINE_string(
        'trainable_scopes', None,
        'Comma-separated list of scopes to filter the set of variables to train.'
        'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
        'ignore_missing_vars', False,
        'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.
    Returns:
        A `Tensor` representing the learning rate.
    Raises:
        ValueError: if
    """
    decay_steps = FLAGS.num_steps_per_decay or \
                  int(num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True,
                                            name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized', FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
        learning_rate: A scalar or `Tensor` learning rate.
    Returns:
        An instance of an optimizer.
    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=FLAGS.adadelta_rho,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=FLAGS.ftrl_learning_rate_power,
                initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                l1_regularization_strength=FLAGS.ftrl_l1,
                l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=FLAGS.momentum,
                name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=FLAGS.rmsprop_decay,
                momentum=FLAGS.rmsprop_momentum,
                epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % FLAGS.train_dir)
        return None
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.
    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
                num_clones=FLAGS.num_clones,
                clone_on_cpu=FLAGS.clone_on_cpu,
                replica_id=FLAGS.task,
                num_replicas=FLAGS.worker_replicas,
                num_ps_tasks=FLAGS.num_ps_tasks)
        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        #########################
        # Initialize parameters #
        #########################

        dataset_dir = FLAGS.dataset_dir
        segment_num = FLAGS.segment_num
        batch_size = FLAGS.batch_size
        epoch_size = FLAGS.epoch_size
        if FLAGS.input_type == 'OpticalFlow':
           channel = 2*FLAGS.num_length
        else:
           channel = 3

        # agg_fn = eval_score.get_agg_fn(FLAGS.agg_fn)

        if FLAGS.max_number_of_epochs is not None:
            max_number_of_steps = math.ceil(FLAGS.max_number_of_epochs*epoch_size / float(batch_size))
        else:
            max_number_of_steps = FLAGS.max_number_of_steps

        num_classes = FLAGS.num_classes


        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
                name=FLAGS.model_name,
                segment_num=segment_num,
                num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                weight_decay=FLAGS.weight_decay,
                is_training=True,
                dropout_keep=FLAGS.dropout_keep,
                channel=channel)

        train_image_size = FLAGS.train_image_size or network_fn.default_image_size
        read_H = FLAGS.read_H or train_image_size
        read_W = FLAGS.read_W or train_image_size

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################

        with tf.device(deploy_config.inputs_device()):

            img_ori_batch, labels, height_read, width_read, _ = preprocessing.train_inputs(
                                                                dataset_dir, segment_num=segment_num,
                                                                batch_size=batch_size,
                                                                num_preprocess_threads=FLAGS.num_preprocessing_threads)

            tf.summary.image('original_image', img_ori_batch[:, :, :, :3])

            img_ori_batch = tf.cast(img_ori_batch, dtype=tf.uint8)
            # print('loading image size:', img_ori_batch)
#           if img_ori_batch.dtype != tf.float32:
#              img_ori_batch = tf.image.convert_image_dtype(img_ori_batch, dtype=tf.float32)

            img_ori_batch = tf.reshape(img_ori_batch, [batch_size*segment_num, height_read, width_read, channel])
            img_ori_batch = tf.image.resize_images(img_ori_batch, [256, 340])

            tf.summary.image('resized_image', img_ori_batch[:, :, :, :3])

            # preprocessing
            # crop the image into network input size
            random_l = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None)
            random_l = random_l[0]
            random_s = tf.random_uniform([1], minval=0, maxval=5, dtype=tf.int32, seed=None, name=None)
            random_s = random_s[0]
            crop_length = tf.convert_to_tensor([240, 224, 192, 168], dtype=tf.int32)
            crop_l = crop_length[random_l]
            crop_size = [batch_size * segment_num, crop_l, crop_l, channel]
            center_h = tf.div(tf.subtract(256, crop_l), 2)
            center_w = tf.div(tf.subtract(340, crop_l), 2)

            crop_start_all = tf.convert_to_tensor([[0, 0, 0, 0],
                                                   [0, 0, 340 - crop_l, 0],
                                                   [0, 256 - crop_l, 0, 0],
                                                   [0, 256 - crop_l, 340 - crop_l, 0],
                                                   [0, center_h, center_w, 0]])

            crop_start = crop_start_all[random_s]

            cropped_img_batch = tf.slice(img_ori_batch, begin=crop_start, size=crop_size)
            tf.summary.image('cropped_resized_image', cropped_img_batch[:,:,:,:3])
            cropped_img_batch = tf.image.resize_images(cropped_img_batch, [train_image_size, train_image_size])

            # raondomly flip left and right
            flip_img_batch = tf.reshape(cropped_img_batch, [batch_size*segment_num*train_image_size,train_image_size,channel])
            flip_img_batch = tf.image.random_flip_left_right(flip_img_batch)
            flip_img_batch = tf.reshape(flip_img_batch, [batch_size*segment_num, train_image_size, train_image_size,channel])

            processed_img_batch = tf.subtract(flip_img_batch, 0.5)
            processed_img_batch = tf.multiply(processed_img_batch, 2.0)
            tf.summary.image('final_distorted_image', processed_img_batch[:,:,:,:3])

            a_out = processed_img_batch
            labels = slim.one_hot_encoding(labels, FLAGS.num_classes - FLAGS.labels_offset)
            b_out = labels
            a_batch, b_batch = a_out, b_out

            batch_queue = slim.prefetch_queue.prefetch_queue(
                    [a_batch, b_batch], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""

            frames, labels = batch_queue.dequeue()
            # frame_batch shape [batch_size, segment_num, height, width, channels]

            frames_group = tf.reshape(frames, [batch_size*segment_num, train_image_size, train_image_size, channel])
            labels = tf.reshape(labels, [batch_size, num_classes])

            logit_out, end_points = network_fn(frames_group)

            logit_out = tf.reshape(logit_out, [batch_size, segment_num, num_classes])
            logit_out = tf.reduce_mean(logit_out, axis=1)
            logit_agg = tf.reshape(logit_out, [batch_size, num_classes])
            # logit_agg = agg_fn(logit_out, k=FLAGS.topK)

            # if 'AuxLogits' in end_points:
            #     aux_logit_agg = tf.reshape(end_points['AuxLogits'], [batch_size, num_classes])
            #     # aux_logit_agg = agg_fn(aux_logit_out, k=FLAGS.topK)
            #     tf.losses.softmax_cross_entropy(
            #             logits=aux_logit_agg, onehot_labels=labels,
            #             label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')

            tf.losses.softmax_cross_entropy(
                    logits=logit_agg, onehot_labels=labels,
                    label_smoothing=FLAGS.label_smoothing, weights=1.0)

            return end_points


        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)

        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################

        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                    FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################

        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(epoch_size, global_step)
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                    opt=optimizer,
                    replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                    variable_averages=variable_averages,
                    variables_to_average=moving_average_variables,
                    replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
                    total_num_replicas=FLAGS.worker_replicas)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))
        # Variables to train.
        variables_to_train = _get_variables_to_train()
        #    and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_dir,
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                init_fn=_get_init_fn(),
                summary_op=summary_op,
                number_of_steps=max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs,
                sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
    tf.app.run()


# frame_batch, label_true_batch = batch_queue.dequeue()
#     # frame_batch shape [batch_size, segment_num, height, width, channels]
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(10):
#         a_val, b_val = sess.run([frame_batch, label_true_batch])
#         print('  b_val:', b_val)
#     # print(a_val, b_val, c_val)
#     print('first batch:')
#     print('  a_val:', a_val.shape)
#     print('  b_val:', b_val)
#     # a_val, b_val= sess.run([a_batch, b_batch,])
#     # print('second batch:')
#     # print('  a_val:',a_val.shape)
#     # print('  b_val:',b_val)
#
#     coord.request_stop()
#     coord.join(threads)
#
# plt.figure(1)
# plt.imshow(a_val[0])
# plt.colorbar()
#
# plt.figure(2)
# plt.imshow(a_val[1])
#
# plt.show()
