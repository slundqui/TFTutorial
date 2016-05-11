import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

inImage = tf.placeholder(tf.float32, shape=[None, 784])
gt = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
est = tf.matmul(inImage, W) + b
loss = tf.reduce_mean(tf.square(gt - est))/2
opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#Calculate accuracy
correct_prediction = tf.equal(tf.argmax(est, 1), tf.argmax(gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_input, batch_gt = mnist.train.next_batch(100)
    sess.run(opt, feed_dict={inImage: batch_input, gt: batch_gt})
    print "Loss on step", i, ":", sess.run(loss, feed_dict={inImage:batch_input, gt:batch_gt})
    print "Accuracy on step", i, ":", sess.run(accuracy, feed_dict={inImage: batch_input, gt: batch_gt})
