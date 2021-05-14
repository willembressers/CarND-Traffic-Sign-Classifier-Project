# python core packages

# 3rd party packages
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
SEED = (1, 2)


def __preprocess(image: tf.Tensor) -> tf.Tensor:
	# normalize between [-1, 1]
	image = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)(image)
	
	# convert RGB to GRAY
	image = tf.image.rgb_to_grayscale(image)
	
	return image


def __augment(image: tf.Tensor) -> tf.Tensor:

	# apply these augmentations on the image (tensor)
	image = tf.image.stateless_random_brightness(image=image, max_delta=0.05, seed=SEED)
	image = tf.image.stateless_random_contrast(image=image, lower=0.7, upper=1.3, seed=SEED)
	image = tf.image.stateless_random_hue(image=image, max_delta=0.08, seed=SEED)
	image = tf.image.stateless_random_saturation(image=image, lower=0.6, upper=1.6, seed=SEED)

	return image


def preprocess(dataset, batch=True, batch_size=32, shuffle=False, augment=False):
	# Apply data augmentation (before grayscaling)
	if augment:
		dataset = dataset.map(lambda image, label: (__augment(image), label), num_parallel_calls=AUTOTUNE)

	# Rescale and grayscale all images
	dataset = dataset.map(lambda image, label: (__preprocess(image), label), num_parallel_calls=AUTOTUNE)

	# shuffle the dataset to ensure a robust network
	if shuffle:
		dataset = dataset.shuffle(buffer_size=int(1e4)).repeat(3)

	# split the dataset into batches
	if batch:
		dataset = dataset.batch(batch_size)

	# Use buffered prefecting on all datasets
	return dataset.prefetch(buffer_size=AUTOTUNE)


def get_input_shape(dataset):
	# get the first batch
	image_batch, labels_batch = next(iter(dataset))

	# get the first image shape
	return image_batch[0].shape