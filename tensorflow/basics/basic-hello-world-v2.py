"""
This is a basic hello-world style program
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds


print("Initializing MNIST dataset")
mnist = tf.keras.datasets.mnist

dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)


print("Loading data")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Building the model by staking layers")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

print("Compiling the model")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model")
model.fit(x_train, y_train, epochs=5)

print("Evaluating the model")
model.evaluate(x_test, y_test)
print("Program completed")
