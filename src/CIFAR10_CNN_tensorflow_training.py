import numpy as np
import os, sys, cPickle

## Hyper-parameters
FC_SIZE1 = 512
# FC_SIZE2 = 256
LR = 1e-4
L2_REG = 0.01
EPOCH = 30
BATCH_SIZE = 200
KERNEL_SIZE1 = 5     ## One side (square)
KERNEL_SIZE2 = 5     ## One side (square)
KERNEL_SIZE3 = 3     ## One side (square)
FEATURE_MAP1 = 32  ## First conv layer feature maps
FEATURE_MAP2 = 32   ## Second conv layer feature maps
FEATURE_MAP3 = 64   ## Second conv layer feature maps
ADAPTIVE_LR = False

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
def unpickle(f):
	fo = open(f, 'rb')
	dictionary = cPickle.load(fo)
	fo.close()
	return dictionary

batch1 = unpickle(repoPath + "data/cifar-10-batches-py/data_batch_1")
batch2 = unpickle(repoPath + "data/cifar-10-batches-py/data_batch_2")
batch3 = unpickle(repoPath + "data/cifar-10-batches-py/data_batch_3")
batch4 = unpickle(repoPath + "data/cifar-10-batches-py/data_batch_4")
batch5 = unpickle(repoPath + "data/cifar-10-batches-py/data_batch_5")
batches = [batch1, batch2, batch3, batch4, batch5]

test_batch = unpickle(repoPath + "data/cifar-10-batches-py/test_batch")

classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
		   5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

training_images = []
training_labels = []
test_images = []
test_labels = []
for batch in batches:
	for image, label in zip(batch['data'], batch['labels']):
		one_hot = np.zeros((10), dtype='float32')
		one_hot[label] = 1.0
		training_images.append(image)
		training_labels.append(one_hot)
training_set = [training_images, training_labels]

for image, label in zip(test_batch['data'], test_batch['labels']):
	one_hot = np.zeros((10), dtype='float32')
	one_hot[label] = 1.0
	test_images.append(image)
	test_labels.append(one_hot)
test_set = [test_images, test_labels]

## Constants
# TRAINING_SIZE = len(training_images)  ## 50000
TRAINING_SIZE = 25000
TEST_SIZE = len(test_images)  ## 10000
TRAINING_BATCH = TRAINING_SIZE / BATCH_SIZE
TEST_BATCH = TEST_SIZE / BATCH_SIZE

####################
## Create Batches ##
####################

training_batches = []   ## [[[batch1_images], [batch1_labels]], [[batch2_images], [batch2_labels]], ... ]
for i in range(0, TRAINING_SIZE, BATCH_SIZE):
	training_batch_image = training_set[0][i:i + BATCH_SIZE]
	training_batch_label = training_set[1][i:i + BATCH_SIZE]
	training_batches.append([training_batch_image, training_batch_label])

test_batches = []
for i in range(0, TEST_SIZE, BATCH_SIZE):
	test_batch_image = test_set[0][i:i + BATCH_SIZE]
	test_batch_label = test_set[1][i:i + BATCH_SIZE]
	test_batches.append([test_batch_image, test_batch_label])

#######################
## Preparing network ##
#######################

import tensorflow as tf
import time
import math

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## Input, output vectors
x = tf.placeholder(tf.float32, [None, 3 * 32 * 32])  ## Here 'None' means that a dimension can be of any length
y_ = tf.placeholder(tf.float32, [None, 10])  ## Correct answers

## 1st conv layer
## [Kernel size1, Kernel size2, first layer channel, second layer channel channel]
with tf.variable_scope("layer1"):
	W_conv1 = tf.get_variable('W_conv1', shape=(KERNEL_SIZE1, KERNEL_SIZE1, 3, FEATURE_MAP1),
								initializer=tf.contrib.layers.xavier_initializer())
	# b_conv1 = tf.get_variable('b_conv1', shape=(FEATURE_MAP1),
	# 							initializer=tf.contrib.layers.variance_scaling_i())
	b_conv1 = tf.Variable(tf.zeros([FEATURE_MAP1]))

## [.., shape1, shape2, channel]
x_image = tf.reshape(x, [-1, 32, 32, 3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## 2nd conv layer
with tf.variable_scope("layer2"):
	W_conv2 = tf.get_variable('W_conv2', shape=(KERNEL_SIZE2, KERNEL_SIZE2, FEATURE_MAP1, FEATURE_MAP2),
								initializer=tf.contrib.layers.xavier_initializer())
	# b_conv2 = tf.get_variable('b_conv2', shape=(FEATURE_MAP2),
	# 							initializer=tf.contrib.layers.xavier_initializer())
	b_conv2 = tf.Variable(tf.zeros([FEATURE_MAP2]))

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## 3rd conv layer
with tf.variable_scope("layer3"):
	W_conv3 = tf.get_variable('W_conv3', shape=(KERNEL_SIZE3, KERNEL_SIZE3, FEATURE_MAP2, FEATURE_MAP3),
								initializer=tf.contrib.layers.xavier_initializer())
	# b_conv3 = tf.get_variable('b_conv3', shape=(FEATURE_MAP2),
	# 							initializer=tf.contrib.layers.xavier_initializer())
	b_conv3 = tf.Variable(tf.zeros([FEATURE_MAP3]))

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

## Fully connected layers

## Image shape halves three times. (32x32) -> (16x16) -> (8x8) -> (4x4) by max_pool_2x2
## Conv. layer does not change image size because of padding='SAME'
with tf.variable_scope("layer4"):
	W_fc1 = tf.get_variable('W_fc1', shape=(4 * 4 * FEATURE_MAP3, FC_SIZE1),
								initializer=tf.contrib.layers.xavier_initializer())
	# b_fc1 = tf.get_variable('b_fc1', shape=(FC_SIZE),
	# 							initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.Variable(tf.zeros([FC_SIZE1]))

h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * FEATURE_MAP3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# with tf.variable_scope("layer5"):
# 	W_fc2 = tf.get_variable('W_fc2', shape=(FC_SIZE1, FC_SIZE2),
# 								initializer=tf.contrib.layers.xavier_initializer())
# 	# b_fc2 = tf.get_variable('b_fc2', shape=(FC_SIZE2),
# 	# 							initializer=tf.contrib.layers.xavier_initializer())
# 	b_fc2 = tf.Variable(tf.zeros([FC_SIZE2]))
#
# h_fc1_flat = tf.reshape(h_fc1, [-1, FC_SIZE1])
# h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)

## Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Softmax output
with tf.variable_scope("layer6"):
	W_fc3 = tf.get_variable('W_fc3', shape=(FC_SIZE1, 10),
								initializer=tf.contrib.layers.xavier_initializer())
	# b_fc3 = tf.get_variable('b_fc3', shape=(10),
	# 							initializer=tf.contrib.layers.xavier_initializer())
	b_fc3 = tf.Variable(tf.zeros([10]))

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

####################
## Cost, accuracy ##
####################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_) +
								L2_REG * tf.nn.l2_loss(W_conv1) +
								L2_REG * tf.nn.l2_loss(b_conv1) +
								L2_REG * tf.nn.l2_loss(W_conv2) +
								L2_REG * tf.nn.l2_loss(b_conv2) +
								L2_REG * tf.nn.l2_loss(W_conv3) +
								L2_REG * tf.nn.l2_loss(b_conv3))
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#############
## Session ##
#############

training_accuracy = 0.0
test_accuracy = 0.0
last_test_accuracy = test_accuracy
t0 = time.time()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	##############
	## Training ##
	##############

	saver.restore(sess, repoPath + "model/CIFAR10_CNN_tensorflow_1/CIFAR10_CNN_tensorflow_1.ckpt")
	print("Model restored.")

	for i in xrange(EPOCH * TRAINING_BATCH):

		j = i % TRAINING_BATCH
		sess.run(train_step, feed_dict={x: training_batches[j][0], y_: training_batches[j][1], keep_prob: 0.75})
		batch_accuracy = sess.run(accuracy, feed_dict={x: training_batches[j][0], y_: training_batches[j][1], keep_prob: 1.0})
		training_accuracy += batch_accuracy

		if (i + 1) / float(TRAINING_BATCH) == (i + 1) / TRAINING_BATCH:
			print "Epoch",  (i + 1) / TRAINING_BATCH, "\n\tTraining accuracy: {0:f}".format(training_accuracy / TRAINING_BATCH)
			training_accuracy = 0.0
			np.random.shuffle(training_batches)

			#############
			## Testing ##
			#############

			test_accuracy = 0.0
			for k in xrange(TEST_BATCH):
				batch_accuracy = sess.run(accuracy, feed_dict={x: test_batches[k][0], y_: test_batches[k][1], keep_prob: 1.0})
				test_accuracy += batch_accuracy

			print "\tTest accuracy: {0:f}".format(test_accuracy / TEST_BATCH)
			np.random.shuffle(test_batches)

			############################
			## Adaptive learning rate ##
			############################

			if ADAPTIVE_LR:
				if test_accuracy > last_test_accuracy and LR >= 1e-3:
					LR += LR * 0.05       ## Increase learning rate by %5 in case of higher accuracy
				elif test_accuracy > last_test_accuracy  and LR < 1e-3:
					LR += LR * 0.90       ## Increase learning rate by %90 when learning is slower
				else:
					LR -= LR * 0.50       ## Half the learning rate in case of lower accuracy
				last_test_accuracy = test_accuracy
				print "\tLearning rate: ", LR

	print "Training time:", time.time() - t0

	save_path = saver.save(sess, repoPath + "model/CIFAR10_CNN_tensorflow_1/CIFAR10_CNN_tensorflow_1.ckpt")
	print("Model saved in file: %s" %save_path)
