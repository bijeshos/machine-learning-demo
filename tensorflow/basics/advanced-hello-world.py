from __future__ import absolute_import, division, print_function

import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import Model

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

# load and prepare MNIST dataset
print("Loading MNIST dataset")
dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


print("Preparing train and test sets")
mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
mnist_test = mnist_test.map(convert_types).batch(32)


# build model using keras model subclassing API
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()

# choose an optimzer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# select metrics to measure the loss and accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# train the model
@tf.function
def train_step(image, label):
    print("inside train_step")
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


# test the model
@tf.function
def test_step(image, label):
    print("inside test_step")
    predictions = model(image)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    test_accuracy(label, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    print("starting epoch", epoch + 1)

    print("performing training")
    for image, label in mnist_train:
        train_step(image, label)

    print("performing testing")
    for test_image, test_label in mnist_test:
        test_step(test_image, test_label)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

print("Program completed")