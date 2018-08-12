import tensorflow as tf

c = tf.constant([[[1, 2], [3, 4], [5, 6]],
                 [[1, 2], [3, 4], [5, 6]]], dtype=tf.float32) #(2, 3, 2)
# conv = tf.layers.conv1d(c, filters=10, kernel_size=1, activation=tf.nn.relu)
conv = tf.layers.conv1d(c, filters=4, kernel_size=3, activation=tf.nn.relu) #(2, 3->1, 2->4)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(c))
print(sess.run(conv))

