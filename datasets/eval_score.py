import tensorflow as tf
import numpy as np



def average_pool3D(score_list, k=None):
    score_list_float = tf.cast(score_list, dtype='float')
    return tf.reduce_mean(score_list_float, axis=1)   # score_list[batch, segment_num, class_num]


def average_pool2D(score_list, k=None):
    score_list_float = tf.cast(score_list, dtype='float')
    return tf.reduce_mean(score_list_float, axis=0)   # score_list[segment_num, class_num]


def max_pool3D(score_list, k=None):
    score_list_float = tf.cast(score_list, dtype='float')
    return tf.reduce_max(score_list_float, axis=1)   # score_list[batch, segment_num, class_num]


# def top_k_pool(score_list, k=1):
#
#     def get_k_segment(segment_list):
#         return np.sort(segment_list, axis=1)[:, -k:, :].mean(axis=1)
#
#     score_list_float = tf.cast(score_list, dtype='float32')
#
#     b1 = tf.py_func(get_k_segment, [score_list_float], tf.float32)
#     return b1


def top_k_pool(score_list, k=1):
    score_list_float = tf.cast(score_list, dtype='float32')
    b = tf.transpose(score_list_float, perm=[0,2,1])
    b = tf.nn.top_k(b, k=k)
    # b = tf.cast(b, dtype='float32')
    b = tf.transpose(b[0], perm=[0,2,1])
    b = tf.reduce_mean(b, axis=1)
    return b


agg_map = {"average" : average_pool3D,
           "max" : max_pool3D,
           "topKpool" : top_k_pool}



def get_agg_fn(agg_name, k=None):
    if agg_name not in agg_map:
        raise ValueError('Name of network unknown %s' % agg_name)
    func = agg_map[agg_name]
    return func



# a = tf.constant([1,2,3,4,5,6,7,8,9,4,5,5,5,4,5,3,1,2])
#
# a = tf.reshape(a, [2,3,3])
#
#
# aa = top_k_pool_test(a)
#
# with tf.Session() as sess:
#     a = a.eval()
#     print(a)
#     b = np.transpose(a, (0,2,1))
#     print(b)
#     print(sess.run(aa))