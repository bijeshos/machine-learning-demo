# A simple program to train a model to convert Celsius to Fahrenheit


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print versions of the libraries used
print('numpy version: {}'.format(np.__version__))
print('tensorflow version: {}'.format(tf.__version__))

# to set logging level at ERROR
# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)

# initialize input data
x_celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# specify output data
y_fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# print input/output data
for i, c in enumerate(x_celsius):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, y_fahrenheit[i]))

# create a dense layer with one neuron and which takes one array
# this is the input layer
layer_0 = tf.keras.layers.Dense(units="1", input_shape=[1])

# add the input layer to the model (in a sequential way)
model = tf.keras.Sequential([layer_0])

# The above two steps can also be specified as follows:
# model = tf.keras.Sequential([
#    tf.keras.layers.Dense(units="1", input_shape=[1])
# ])

# compile the model with loss function MSE and Adam optimizer with a learning rate 0.1
# learning rates could be anywhere between 0.1 to 0.001, depending on the problem
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# fit (train) the model using input and output values. This will be done for 500 iterations
history = model.fit(x_celsius, y_fahrenheit, epochs=500)

# use below statement to disable verbosity
# history = model.fit(x_celsius,y_fahrenheit,epochs=500, verbose=False)

# plot the progress of loss magnitude against the epochs.
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

# predict the value for a test value
print('predicting value for 100.0:')
print(model.predict([100.0]))

# print the values used by the network
print("These are the layer variables: {}".format(layer_0.get_weights()))

# The same problem can also be tackled with a multi-layered neural network.
# Following is an approach with 3 layers
layer_0 = tf.keras.layers.Dense(units=4, input_shape=[1])
layer_1 = tf.keras.layers.Dense(units=4)
layer_2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([layer_0, layer_1, layer_2])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(x_celsius, y_fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

print(model.predict([100.0]))

print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the layer_0 variables: {}".format(layer_0.get_weights()))
print("These are the layer_1 variables: {}".format(layer_1.get_weights()))
print("These are the layer_2 variables: {}".format(layer_2.get_weights()))
