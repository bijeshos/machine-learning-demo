from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# list available datasets
print(tfds.list_builders())

# construct a tf.data.Dataset
#dataset, info = tfds.load(name="mnist", split="train", with_info=True)

# using S3 API splits
train_ds, test_ds = tfds.load('mnist:3.*.*', split=['train', 'test'], batch_size=-1)

# train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
numpy_ds = tfds.as_numpy(train_ds)
numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]

print("Label: %d" % numpy_labels[0])
plt.imshow(numpy_images[0][:, :, 0], cmap=plt.get_cmap("gray"))
plt.show()

print("Label: %d" % numpy_labels[10])
plt.imshow(numpy_images[10][:, :, 0], cmap=plt.get_cmap("gray"))
plt.show()



# for mnist_train_example in train_ds.take(1):
#   image, label = mnist_train_example["image"], mnist_train_example["label"]
#
#   plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
#   plt.show()
#   print("Label: %d" % label.numpy())


# for mnist_test_example in test_ds.take(1):  # Only take a single example
#   image, label = mnist_test_example["image"], mnist_test_example["label"]
#
#   plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
#   plt.show()
#   print("Label: %d" % label.numpy())


# dataset, info = tfds.load(name="mnist", with_info=True)
#
# train_data, test_data = dataset['train'], dataset[test]
# info.features['label'].num_classes
# info.splits['train'].num_classes
#
# dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# for features in dataset.take(1):
#     image, label = features["image"], features["label"]
#     print(image, label)


print("Program completed")
