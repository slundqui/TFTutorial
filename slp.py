import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#Get mnist object
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Input image is 28 x 28
#Placeholders
inImage = tf.placeholder(tf.float32, shape=[None, 784], name="inImage")
gt = tf.placeholder(tf.float32, shape=[None, 10], name="gt")

#Variables to learn
W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name = "B")

#Operations
with tf.name_scope("est"):
    est = tf.matmul(inImage, W) + b
    #est = tf.nn.softmax(tf.matmul(inImage, W) + b)
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(gt - est))/2
    #loss = tf.reduce_mean(-tf.reduce_sum(gt * tf.log(est),
    #    reduction_indices=[1]))
with tf.name_scope("optimizer"):
    opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#Calculate accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(est, 1), tf.argmax(gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Tensorboard visualizations
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.histogram('inImage', inImage)
tf.summary.histogram('gt', gt)
tf.summary.histogram('est', est)
tf.summary.histogram('W', W)
tf.summary.histogram('b', b)
tf.summary.image("weights", tf.reshape(tf.transpose(W), (10, 28, 28, 1)),
        max_outputs=10)

#Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Summary writer
mergedSummary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("output/train", sess.graph)
test_writer = tf.summary.FileWriter("output/test")

#Evaluation
for i in range(1000):
    print("Timestep: ", i)
    #Get input and gt of batch size 100
    batch_input, batch_gt = mnist.train.next_batch(256)
    #Run optimizer
    sess.run(opt, feed_dict={inImage: batch_input, gt: batch_gt})

    #Run summary on test every 10 timesteps
    if(i % 10 == 0):
        #Run summary on train
        train_summary = sess.run(mergedSummary,
                feed_dict={inImage: batch_input, gt: batch_gt})
        train_writer.add_summary(train_summary, i)
        #Run summary on test
        test_summary = sess.run(mergedSummary,
                feed_dict={inImage: mnist.test.images, gt: mnist.test.labels})
        test_writer.add_summary(test_summary, i)
