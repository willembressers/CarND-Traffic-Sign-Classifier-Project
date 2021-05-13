# python core packages

# 3rd party packages
# import numpy as np
import tensorflow as tf


def dataset(model, dataset, class_names):
	predictions = model.predict(dataset)

	probabilities = []
	labels = []

	for logits in predictions:
		# get the class_id with the highest probability
		class_id = tf.argmax(logits).numpy()

		# get the probability
		probabilities.append(tf.nn.softmax(logits)[class_id].numpy())
		
		# get the label
		labels.append(class_names[class_id])

	return labels, probabilities


	# # add a softmax layer to the model
	# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

	# # predict the class probability of the new images
	# predictions = probability_model.predict(dataset)

	# return tf.argmax(predictions, axis=1)
	# return tf.math.argmax(predictions, axis=1)
