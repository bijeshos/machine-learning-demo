"""
This is a sample program to perform text classification using IMDB data set
"""
# reference: https://www.tensorflow.org/tutorials


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

print("----------------------------------")
print("Tensorflow version: ", tf.__version__)
print("----------------------------------")

print("Importing IMDB dataset")
imdb = keras.datasets.imdb

print("Loading datasets to NumPy arrays")
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore data
print("Training data: entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print("First training review: ", train_data[0])

print("Length of first review: ", len(train_data[0]))
print("Length of second review: ", len(train_data[1]))

print("Converting the integers back to words")

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print("Decoded review: ", decode_review(train_data[0]))

print("Preparing the data")
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print("Length of first review: ", len(train_data[0]))
print("Length of second review: ", len(train_data[1]))

print("Training data 0: ", train_data[0])

print("Building the model")

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

print("Configuring optimizer")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Creating a validation set")
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

print("Training the model")
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
print("Evaluating the model")
results = model.evaluate(test_data, test_labels)

print("Evaluation Results: ", results)

print("Creating a graph of accuracy and loss over time")
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print("----------------------------------")
print("Program completed")
print("----------------------------------")
