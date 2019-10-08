"""
This is a sample program to perform image classification using Fashion MNIST image data set
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime

print("----------------------------------")
print("Tensorflow version: ", tf.__version__)
print("----------------------------------")

print("Importing Fashion MNIST dataset (Keras)")
keras_fashion_mnist = keras.datasets.fashion_mnist
(train_keras_images, train_keras_labels), (test_keras_images, test_keras_labels) = keras_fashion_mnist.load_data()

print("Importing Fashion MNIST dataset (tfds)")
train_tfds, test_tfds = tfds.load('mnist:3.*.*', split=['train', 'test'], batch_size=-1)

train_tfds_nmpy = tfds.as_numpy(train_tfds)
train_tfds_images, train_tfds_labels = train_tfds_nmpy["image"], train_tfds_nmpy["label"]

# print(train_tfds_images.shape[0])
# print(train_tfds_images.shape[1])
# print(train_tfds_images.shape[2])
# print(train_tfds_images.shape[3])

train_tfds_images = train_tfds_images.reshape(train_tfds_images.shape[0], train_tfds_images.shape[1], train_tfds_images.shape[2])

print(train_tfds_images.shape)

test_numpy_ds = tfds.as_numpy(test_tfds)
test_tfds_images, test_tfds_labels = test_numpy_ds["image"], test_numpy_ds["label"]
test_tfds_images = test_tfds_images.reshape(test_tfds_images.shape[0], test_tfds_images.shape[1], test_tfds_images.shape[2])

print("Storing class names")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("----------------------------------")
print("Exploring data")

print("keras: Training image shape/format: ", train_keras_images.shape)
print("keras: Training labels shape/format: ", train_keras_labels.shape)
# print("keras: Number of training labels: ", len(train_keras_labels))
# print("keras: Training labels: ", train_keras_labels)

print("keras: Test image shape/format: ", test_keras_images.shape)
print("keras: Test labels shape/format: ", test_keras_images.shape)
# print("keras: Number of test labels: ", len(test_keras_images))

print("tfds: Training image shape/format: ", train_tfds_images.shape)
print("tfds: Training labels shape/format: ", train_tfds_labels.shape)
# print("tfds: Number of training labels: ", len(train_tfds_labels))
# print("tfds: Training labels: ", train_tfds_labels)

print("tfds: Test image shape/format: ", test_tfds_images.shape)
print("tfds: Test labels shape/format: ", test_tfds_labels.shape)
# print("tfds: Number of test labels: ", len(test_tfds_labels))

print("Pre-processing data for display. Close popup to continue ...")
plt.figure()
plt.imshow(train_tfds_images[0])
#plt.imshow(train_tfds_images[0][:, :, 0], cmap=plt.get_cmap("gray"))
plt.colorbar()
plt.grid(False)
plt.show()

print("Scaling values")
train_tfds_images = train_tfds_images / 255.0

test_tfds_images = test_tfds_images / 255.0

print("Displaying first 25 images for verification. Close popup to continue ...")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_tfds_images[i], cmap=plt.cm.binary)
    # plt.imshow(train_keras_images[i][:, :, 0], cmap=plt.get_cmap("gray"))
    plt.xlabel(class_names[train_tfds_labels[i]])
plt.show()

print("----------------------------------")
print("Building the model")

print("Setting up the layers")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

print("Compiling the model")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print("Training the model")
# model.fit(train_images, train_labels, epochs=5)

model.fit(train_tfds_images,
          train_tfds_labels,
          epochs=5,
          validation_data=(test_tfds_images, test_tfds_labels),
          callbacks=[tensorboard_callback])

print("Evaluating accuracy")
test_loss, test_acc = model.evaluate(test_tfds_images, test_tfds_labels)

print("Test accuracy:", test_acc)

print("----------------------------------")
print("Making predictions")
predictions = model.predict(test_tfds_images)

print("Prediction 0: ", predictions[0])

print("----------------------------------")
print("Label with highest confidence: ", np.argmax(predictions[0]))

print("Test Label 0: ", test_tfds_labels[0])
print("----------------------------------")
print("Plotting as a graph. Close popup to continue ...")


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


print("Showing 0th image. Close popup to continue ...")
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_tfds_labels, test_tfds_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_tfds_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_tfds_labels, test_tfds_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_tfds_labels)
plt.show()

print("----------------------------------")
print("Plotting the first X test images, their predicted labels, and the true labels.")
print("Coloring correct predictions in blue and incorrect predictions in red.")
print("Close popup to continue ...")
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_tfds_labels, test_tfds_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_tfds_labels)
plt.show()

print("----------------------------------")
print("Using trained model to make prediction")

print("Grabbing an image from the test dataset.")
img = test_tfds_images[0]

print("Test image shape: ", img.shape)

print("Adding the image to a batch where it's the only member.")
img = (np.expand_dims(img, 0))

print("Test image shape: ", img.shape)

print("----------------------------------")
print("Predicting correct label for the image")
predictions_single = model.predict(img)
print("Single Prediction: ", predictions_single)

plot_value_array(0, predictions_single, test_tfds_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
# todo: check this later
# file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)

print("----------------------------------")
print("Program completed")
print("----------------------------------")
