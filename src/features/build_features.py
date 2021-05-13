# python core packages

# 3rd party packages
import tensorflow as tf


# AUTOTUNE = tf.data.AUTOTUNE


# def prepare(dataset, batch_size=32, shuffle=False, augment=False):
#   # Resize and rescale all datasets
#   dataset = dataset.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

#   if shuffle:
#     dataset = dataset.shuffle(buffer_size=int(1e4))

#   # Batch all datasets
#   dataset = dataset.batch(batch_size)

#   # Use data augmentation only on the training set
#   if augment:
#     dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

#   # Use buffered prefecting on all datasets
#   return dataset.prefetch(buffer_size=AUTOTUNE)


def __preprocess(image, label):    
    # normalize between [0, 1]
    image = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image)
    # image = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)(image)
    
    # convert RGB to GRAY
    image = tf.image.rgb_to_grayscale(image)
    
    # return 
    return image, label


def preprocess(dataset):
    # Apply the processing function to all tensors
    return dataset.map(lambda image, label: __preprocess(image, label), num_parallel_calls=tf.data.AUTOTUNE)


def get_input_shape(dataset):
	# get the first batch
	image_batch, labels_batch = next(iter(dataset))

	# get the first image shape
	return image_batch[0].shape


# def augmentation(dataset):
# 	augmentations = [color, rotate]

# 	for augmentation_function in augmentations:
# 		dataset = dataset.map(lambda image, label: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: augmentation_function(image), lambda: image, label), num_parallel_calls=tf.data.AUTOTUNE)

# 	return dataset.map(lambda x: tf.clip_by_value(x, 0, 1))


# def color(x: tf.Tensor) -> tf.Tensor:
#     x = tf.image.random_hue(x, 0.08)
#     x = tf.image.random_saturation(x, 0.6, 1.6)
#     x = tf.image.random_brightness(x, 0.05)
#     x = tf.image.random_contrast(x, 0.7, 1.3)
#     return x


# def rotate(x: tf.Tensor) -> tf.Tensor:
#     return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))