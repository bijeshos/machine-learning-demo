# reference: https://www.tensorflow.org/tutorials

import tensorflow as tf
import tensorflow_datasets as tfds

# list available datasets
print(tfds.list_builders())

# construct a tf.data.Dataset
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

for features in dataset.take(1):
    image, label = features["image"], features["label"]
    print(image, label)
print("Program completed")
