import numpy as np
from keras.models import load_model
from keras import backend as K
import cPickle, os, sys

BATCH_SIZE = 100
MODEL_PATH = "../model/CIFAR10_CNN_keras/CIFAR10_CNN_keras_6-3.h5"

## Load model
model = load_model(MODEL_PATH)
print("Model restored.")

## Read test data

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
def unpickle(f):
	fo = open(f, 'rb')
	dictionary = cPickle.load(fo)
	fo.close()
	return dictionary

test_batch = unpickle(repoPath + "data/cifar-10-batches-py/test_batch")

classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
		   5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

test_images = []
test_labels = []
for image, label in zip(test_batch['data'], test_batch['labels']):
	one_hot = np.zeros((10), dtype='float32')
	one_hot[label] = 1.0
	test_images.append(image)
	test_labels.append(one_hot)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

## Reshape images
if K.image_data_format() == 'channels_first':
	test_images = test_images.reshape(test_images.shape[0], 3, 32, 32)
	input_shape = (3, 32, 32)
else:
	test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
	input_shape = (32, 32, 3)

# ## See first n-image with their labels
# import matplotlib.pyplot as plt
# n = 5
# i = 1
# for image, label in zip(test_images[:n], test_labels[:n]):
# 	plt.subplot(1, n, i)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.title(classes[label.nonzero()[0][0]])  ## Label
# 	plt.imshow(image.reshape((3, 32, 32)).transpose(1, 2, 0), cmap=plt.cm.Greys)
# 	i += 1
# plt.show()

## Evaluate test data
loss, accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, verbose=1, sample_weight=None)
print "\nLoss:", loss
print "Accuracy:", accuracy
