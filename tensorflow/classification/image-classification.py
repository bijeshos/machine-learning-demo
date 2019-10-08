"""
This is a sample program to perform image classification using Fashion MNIST image data set
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("----------------------------------")
print("Tensorflow version: ", tf.__version__)
print("----------------------------------")
print("Importing Fashion MNIST dataset")
fashion_mnist = keras.datasets.fashion_mnist

print("Loading datasets to NumPy arrays")
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Storing class names")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("----------------------------------")
print("Exploring data")

print("Training image shape/format: ", train_images.shape)

print("Number of training labels: ", len(train_labels))

print("Training labels: ", train_labels)

print("Test image shape/format: ", test_images.shape)

print("Number of test labels: ", len(test_labels))

print("Pre-processing data for display. Close popup to continue ...")
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("Scaling values")
train_images = train_images / 255.0

test_images = test_images / 255.0

print("Displaying first 25 images for verification. Close popup to continue ...")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
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

print("Training the model")
model.fit(train_images, train_labels, epochs=5)

print("Evaluating accuracy")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)

print("----------------------------------")
print("Making predictions")
predictions = model.predict(test_images)

print("Prediction 0: ", predictions[0])

print("----------------------------------")
print("Label with highest confidence: ", np.argmax(predictions[0]))

print("Test Label 0: ", test_labels[0])
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
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
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
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

print("----------------------------------")
print("Using trained model to make prediction")

print("Grabbing an image from the test dataset.")
img = test_images[0]

print("Test image shape: ", img.shape)

print("Adding the image to a batch where it's the only member.")
img = (np.expand_dims(img, 0))

print("Test image shape: ", img.shape)

print("----------------------------------")
print("Predicting correct label for the image")
predictions_single = model.predict(img)
print("Single Prediction: ", predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

print("----------------------------------")
print("Program completed")
print("----------------------------------")
