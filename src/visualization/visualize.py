# python core packages
import random

# 3rd party packages
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_images(X_train, y_train, n_train, class_names, title=None, n_rows = 2, n_columns = 6):

	# generate n random integers from the trainingsset
	integers = [random.randint(0, n_train) for p in range(0, (n_columns * n_rows))]

	# show the images
	fig, axs = plt.subplots(n_rows, n_columns, figsize=(23, 7))
	fig.suptitle(title)
	for index, integer in enumerate(integers):
		class_id = y_train[integer]
		
		ax = plt.subplot(n_rows, n_columns, index + 1)
		plt.imshow(X_train[integer])
		plt.title(f'{class_names[class_id]} ({class_id})')
		plt.axis("off")


def image_grid(dataset, class_names, title=None, n_rows = 2, n_columns = 6):
	fig, axs = plt.subplots(n_rows, n_columns, figsize=(23, 7))
	fig.suptitle(title)

	# get a batch    
	image_batch, labels_batch = next(iter(dataset))
	batch_size = image_batch.shape[0]
	index = 0

	# loop over the rows and columns
	for y in range(n_rows):
		for x in range(n_columns):

			# Quit if there are no more images
			if index >= batch_size:
				return

			# get the class_id and the image
			class_id = int(labels_batch[index].numpy())
			image = image_batch[index].numpy()

			# handle RGB and grayscale images accordingly
			cmap = 'gray' if image.shape[2] == 1 else 'viridis'

			# show the image
			axs[y, x].imshow(image, cmap=cmap)
			axs[y, x].set_title(f'{class_names[class_id]} ({class_id})')
			axs[y, x].axis("off")

			# get the next image
			index += 1


def class_distribution(y_train, class_names, title='', threshold=1000):
	# count per value
	values, counts = np.unique(y_train, return_counts=True)

	# map the values to labels
	labels = [class_names[value] for value in values]

	# create a dataframe (so i can sort) and then plot the value
	df = pd.DataFrame({'labels':labels, 'counts':counts}).sort_values('counts', ascending=True).set_index('labels')

	# define a subplot
	fig, ax = plt.subplots(figsize=(20, 15))
	
	# plot the bars
	bars = ax.barh(df.index, df['counts'])
	
	# draw the vertical threshold line
	ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

	# remove the borders
	for spine in plt.gca().spines.values():
		spine.set_visible(False)

	# loop over the bars
	for bar in bars:
		# get the bar value
		label = bar.get_width()
		
		# determine the label y position
		label_y_pos = bar.get_y() + bar.get_height() / 2
		
		# add the label
		ax.text(label, label_y_pos, s=f'{label:.0f}', va='center', ha='right', fontsize=15, color='white')
		
		# color the bars 
		if label > threshold:
			bar.set_color('green')
		else:
			bar.set_color('orange')

	plt.title(title)
	plt.xlabel('Nr images')
	plt.ylabel('Class label')
	plt.legend()


def history(history, threshold):
	plt.figure(figsize=(22, 10))

	# summarize history for accuracy
	plt.subplot(3, 1, 1)
	plt.plot(history.history['accuracy'], label='Training')
	plt.plot(history.history['val_accuracy'], label='Validation')
	plt.axhline(y=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')
	plt.legend(loc='lower right')
	plt.title('Training & Validation history')
	plt.ylabel('Accuracy')
	plt.ylim([0,1.0])

	# summarize history for loss
	plt.subplot(3, 1, 2)
	plt.plot(history.history['loss'], label='Training')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.legend(loc='upper right')
	plt.ylabel('Loss')
	plt.ylim([0,1.0])

	# summarize history for learning rate
	plt.subplot(3, 1, 3)
	plt.plot(history.history['lr'])
	plt.ylabel('Learning rate')
	plt.xlabel('Epoch')


def evaluations(df, threshold = 0.9):
	# define a subplot
	fig, ax = plt.subplots(figsize=(20, 5))
	
	# plot the bars
	bars = ax.barh(df.index, df['accuracy'])
	
	# draw the vertical threshold line
	ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

	# remove the borders
	for spine in plt.gca().spines.values():
		spine.set_visible(False)

	# loop over the bars
	for bar in bars:
		# get the bar value
		label = bar.get_width()
		
		# determine the label y position
		label_y_pos = bar.get_y() + bar.get_height() / 2
		
		# add the label
		ax.text(label, label_y_pos, s=f'{label:.3f}', va='center', ha='right', fontsize=15, color='white')
		
		# color the bars 
		if label > threshold:
			bar.set_color('green')
		else:
			bar.set_color('orange')

	plt.title('Accuracy per dataset')
	plt.xlabel('Accuracy')
	plt.ylabel('Dataset')
	plt.legend()


def confusion_matrix(y_true, y_pred):
	# get the class_names as a list (from the dict)
	display_labels = list(np.unique(np.concatenate((y_true, y_pred))))

	# plot a figure
	fig, axs = plt.subplots(ncols=2, figsize=(20, 7))
	fig.suptitle('Confusion matrices')

	# loop over the 2 options
	for title, normalize, ax in [("Absolute values", None, axs[0]),("Normalized values", 'true', axs[1])]:

		# calculate the confusion matrix (add labels otherwise the matrix isn't what you expect)
		confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize=normalize, labels=display_labels)

		# display the confusion matrix
		disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)

		# add the confusion matrix to the plot
		disp.plot(cmap=plt.cm.Blues, ax=ax)

		# specify a title
		disp.ax_.set_title(title)

		# rotate the x labels (and allign them right)
		for label in ax.get_xticklabels():
			label.set_rotation(45)
			label.set_ha('right')

	plt.tight_layout()