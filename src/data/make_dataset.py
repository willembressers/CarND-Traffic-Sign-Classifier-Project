# python core packages

# 3rd party packages
import tensorflow as tf

def from_tensor_slices(X, y, batch_size=32, shuffle=False, buffer_size=1000):
	# create a dataset
	dataset = tf.data.Dataset.from_tensor_slices((X, y))

	# shuffle the dataset
	if shuffle:
		dataset = dataset.shuffle(buffer_size).repeat()

	# generate batches
	dataset = dataset.batch(batch_size)

	return dataset