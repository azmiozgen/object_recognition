from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import random
import os, sys, cPickle

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
training_set = []
test_images = []
test_labels = []
for batch in batches:
	for image, label in zip(batch['data'], batch['labels']):
		one_hot = np.zeros((10), dtype='float32')
		one_hot[label] = 1.0
		training_images.append(image)
		training_labels.append(one_hot)
		training_set.append((one_hot, image))

for image, label in zip(test_batch['data'], test_batch['labels']):
	one_hot = np.zeros((10), dtype='float32')
	one_hot[label] = 1.0
	test_images.append(image)
	test_labels.append(one_hot)

training_images = np.array(training_images)
training_labels = np.array(training_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

## Reshape
training_images = training_images.reshape(training_images.shape[0], 3, 32, 32)

# Convert from int to float
training_images = training_images.astype('float32')

## See random n-digit with their labels
import matplotlib.pyplot as plt
n = 5
random.shuffle(training_set)
fig = plt.figure()
for i, digit in enumerate(training_set[:n]):
	plt.subplot(1, n, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.title(classes[digit[0].nonzero()[0][0]])  ## Label
	plt.imshow(digit[1].reshape((3, 32, 32)).transpose(1, 2, 0), cmap=plt.cm.Greys)
plt.show()

# Define data preparation
datagen = ImageDataGenerator(featurewise_center=False,
							samplewise_center=False,
							featurewise_std_normalization=False,
							samplewise_std_normalization=False,
							zca_whitening=False,
							rotation_range=30.,
							width_shift_range=0.2,
							height_shift_range=0.2,
							shear_range=0.05,
							zoom_range=0.1,
							channel_shift_range=0.0,
							fill_mode='nearest',
							cval=0.0,
							horizontal_flip=True,
							vertical_flip=False,
							rescale=None)

# Fit parameters from data
datagen.fit(training_images)

# BATCH_SIZE = 200
# counter = 1
# # Configure batch size and retrieve one batch of images
# with open("../data/extended2_train.csv", mode="w") as f:
# 	for training_images, training_labels in datagen.flow(training_images, training_labels, batch_size=BATCH_SIZE, shuffle=True):
# 		for i in xrange(BATCH_SIZE):
# 			f.write(str(training_labels[i].nonzero()[0][0]) + ",")
# 			digit = training_images[i].reshape(3 * 32 * 32)
# 			digit = np.rint(digit).astype(np.int32)
# 			for pixel in digit:
# 				f.write(str(pixel) + ",")
# 			f.seek(-1, os.SEEK_END)
# 			f.truncate()
# 			f.write("\n")
# 		print "Batch:", counter
# 		counter += 1
# 		if counter == (50000 / BATCH_SIZE) + 1:
# 			break
