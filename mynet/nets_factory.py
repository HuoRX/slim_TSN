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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf


from mynet.inception_v3_frozen_BN import inception_v3_frozen_BN
from mynet.inception_v2_frozen_BN import inception_v2_frozen_BN
# from mynet.inception_v3_frozen_BN_2 import inception_v3_frozen_BN_arg_scope
# from mynet.inception_v3_frozen_BN_2 import inception_v3_frozen_BN_base

slim = tf.contrib.slim


def inception_frozen_BN_arg_scope(weight_decay=0.0005,
                                    use_batch_norm=True,
                                    batch_norm_decay=1.0,
                                    batch_norm_epsilon=0.001):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # collection containing update_ops.
      'is_training': False,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc



def get_network_fn(name='inception_v3_frozen_BN', num_classes=101,channel=3,
                   weight_decay=0.0005, is_training=False, segment_num=5,
                   dropout_keep=0.8):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  networks_map = {'inception_v3_frozen_BN': inception_v3_frozen_BN,
                  'inception_v2_frozen_BN': inception_v2_frozen_BN}

  arg_scope = inception_frozen_BN_arg_scope(weight_decay=weight_decay)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, segment_num=segment_num, dropout_keep_prob=dropout_keep)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
