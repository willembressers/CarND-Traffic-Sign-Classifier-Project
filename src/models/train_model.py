# python core packages

# 3rd party packages
import tensorflow as tf


def lenet_architecture(input_shape, n_classes):
	model = tf.keras.Sequential([

		# Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
		tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
		tf.keras.layers.AveragePooling2D(),

		#  Layer 2: Convolutional. Output = 10x10x16.
		tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
		tf.keras.layers.AveragePooling2D(),

		# Flatten. Input = 5x5x16. Output = 400.
		tf.keras.layers.Flatten(),

		# Layer 3: Fully Connected. Input = 400. Output = 120.
		tf.keras.layers.Dense(units=120, activation='relu'),
		tf.keras.layers.Dropout(0.1),

		# Layer 4: Fully Connected. Input = 120. Output = 84.
		tf.keras.layers.Dense(units=84, activation='relu'),
		tf.keras.layers.Dropout(0.1),

		#  Layer 5: Fully Connected. Input = 84. Output = 10.
		tf.keras.layers.Dense(units=n_classes)
	])

	# compile the model
	model.compile(
		optimizer = 'adam',
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics = ['accuracy']
	)

	return model

def train(model, train_dataset, validation_dataset, n_train, batch_size, epochs=50):
	steps_per_epoch = n_train//batch_size

	# callbacks
	callbacks = [
	    # prevent overfitting
	    tf.keras.callbacks.EarlyStopping(patience=10),
	    
	    # decrease learning rate
	    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4),
	]

	# train the model
	history = model.fit(
	    train_dataset, 
	    epochs=epochs,
	    validation_data=validation_dataset,
	    callbacks=callbacks,
	    steps_per_epoch=steps_per_epoch
	)

	return history