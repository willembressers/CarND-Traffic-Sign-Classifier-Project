# python core packages

# 3rd party packages
import tensorflow as tf


def lenet_architecture(input_shape, n_classes):
	return tf.keras.Sequential([

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


def compile_and_fit(model, train_dataset, validation_dataset, class_weight=None, learning_rate=0.001, epochs=50):

	# compile the model
	model.compile(
		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), 
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics = ['accuracy']
	)

	# train the model
	return model.fit(
		train_dataset, 
		epochs=epochs,
		validation_data=validation_dataset,
		callbacks=[
			tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
			tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
		],
		class_weight=class_weight,
	)