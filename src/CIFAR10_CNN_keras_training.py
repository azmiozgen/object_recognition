import numpy as np
import random
import os, sys, cPickle

## Hyper-parameters
FC_SIZE1 = 512
FC_SIZE2 = 256
LR = 1e-5
L2_REG = 0.001
EPOCH = 4
BATCH_SIZE = 100
MOMENTUM = 0.9
DECAY = 1e-4		 ## lr = self.lr * (1. / (1. + self.decay * self.iterations))
KERNEL_SIZE1 = 3     ## One side (square)
# KERNEL_SIZE2 = 5
FEATURE_MAP1 = 48
FEATURE_MAP2 = 96
FEATURE_MAP3 = 192
DROPOUT1 = 0.25
DROPOUT2 = 0.50
VALIDATION = 0.0		## When 0.0, super learning is active (Real-time data augmentation)
SUPER = False

RESTORE = True

MODEL_PATH = "../model/CIFAR10_CNN_keras/CIFAR10_CNN_keras_6-3.h5"

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

TRAINING_SIZE = training_images.shape[0] * (1 - VALIDATION)
VALIDATION_SIZE = training_images.shape[0] - TRAINING_SIZE

# ## See random n-digit with their labels
# import matplotlib.pyplot as plt
# n = 5
# random.shuffle(training_set)
# fig = plt.figure()
# for i, im in enumerate(training_set[:n]):
# 	plt.subplot(1, n, i + 1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.title(classes[im[0].nonzero()[0][0]])  ## Label
# 	plt.imshow(im[1].reshape((3, 32, 32)).transpose(1, 2, 0))
# plt.show()

## Preparing network

from keras import backend as K
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
from keras.initializers import glorot_uniform
from keras.regularizers import l1_l2, l1, l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

## Reshape images
if K.image_data_format() == 'channels_first':
	training_images = training_images.reshape(training_images.shape[0], 3, 32, 32)
	input_shape = (3, 32, 32)
else:
	training_images = training_images.reshape(training_images.shape[0], 32, 32, 3)
	input_shape = (32, 32, 3)


## Shuffle training set
training_images, training_labels = shuffle(training_images, training_labels, random_state=np.random.choice(range(EPOCH)))

if RESTORE and os.path.isfile(MODEL_PATH):
	model = load_model(MODEL_PATH)
	print("Model restored.")
else:
	model = Sequential()
	model.add(Conv2D(FEATURE_MAP1, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', input_shape=input_shape,
					kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(Conv2D(FEATURE_MAP1, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(DROPOUT1))
	model.add(Conv2D(FEATURE_MAP2, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(Conv2D(FEATURE_MAP2, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(DROPOUT1))
	model.add(Conv2D(FEATURE_MAP3, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(Conv2D(FEATURE_MAP3, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l2(L2_REG)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(DROPOUT1))

	model.add(Flatten())
	model.add(Dense(FC_SIZE1, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dropout(DROPOUT2))
	model.add(Dense(FC_SIZE2, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(L2_REG)))
	model.add(Dropout(DROPOUT2))
	model.add(Dense(10, activation='softmax'))

	sgd = SGD(lr=LR, momentum=MOMENTUM, decay=DECAY, nesterov=True)
	adam = Adam(lr=LR)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])


###########
## Train ##
###########


## See model summary
print model.summary()

# ## Set new LR and decay etc.
K.set_value(model.optimizer.lr, LR)
K.set_value(model.optimizer.decay, DECAY)
print "LR:",       K.get_value(model.optimizer.lr)
print "DECAY:",    K.get_value(model.optimizer.decay)


## Normal learning mode
if VALIDATION != 0.0 or (not SUPER):
	model.fit(training_images, training_labels, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=VALIDATION, shuffle=True)

## Real-time data augmentation when VALIDATION = 0.0 (Super learning mode)
elif VALIDATION == 0.0 and SUPER:

	# Define data preparation
	datagen = ImageDataGenerator(featurewise_center=False,
								samplewise_center=False,
								featurewise_std_normalization=False,
								samplewise_std_normalization=False,
								zca_whitening=False,
								rotation_range=0.0,
								width_shift_range=0.0,
								height_shift_range=0.0,
								shear_range=0.0,
								zoom_range=0.0,
								channel_shift_range=0.0,
								fill_mode='nearest',
								cval=0.0,
								horizontal_flip=True,
								vertical_flip=False,
								rescale=None)

	datagen.fit(training_images)
	model.fit_generator(datagen.flow(training_images, training_labels, batch_size=BATCH_SIZE), steps_per_epoch=TRAINING_SIZE / BATCH_SIZE, epochs=EPOCH)

	# for e in range(EPOCH):
	# 	print('Epoch', e)
	# 	batches = 0
	# 	for training_image, training_label in datagen.flow(training_images, training_labels, batch_size=BATCH_SIZE):
	# 		model.fit(training_image, training_label)
	# 		batches += 1
	# 		if batches >= TRAINING_SIZE / BATCH_SIZE:
	# 			break

# MODEL_PATH = "../model/CIFAR10_CNN_keras/CIFAR10_CNN_keras_6-3_mod.h5"
model.save(MODEL_PATH)
print("Model saved.")

