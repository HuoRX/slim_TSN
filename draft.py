import tensorflow as tf

a = tf.constant([1,2,3,4,5,6,7,8,9])

b = tf.reshape(a,[3,3])

with tf.Session() as sess:
    out = sess.run(b)
    print(out)