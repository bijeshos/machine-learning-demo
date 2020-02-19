"""
This is a sample program to perform classification on structured data from Cleveland Clinic Foundation for Heart Disease
"""
# reference: https://www.tensorflow.org/tutorials


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Import Cleveland Clinic Foundation for Heart Disease CSV
print("Using Pandas to create a dataframe")
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)

print("Created dataframe is:")
print(dataframe.head())

print("Split the dataframe into train, validation, and test")
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print("Train examples: ", len(train))
print("Validation examples: ", len(val))
print("Test examples: ", len(test))

print("Creating an input pipeline using tf.data")


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print("Understanding the input pipeline")
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch)

print("Demonstrating several types of feature column")
print("We will use this batch to demonstrate several types of feature columns")
example_batch = next(iter(train_ds))[0]


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


print("Numeric columns")
age = feature_column.numeric_column("age")
demo(age)

print("Bucketized columns")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

print("Categorical columns")
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

print("Embedding columns")
print("Notice the input to the embedding column is the categorical column")
print("we previously created")
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

print("Hashed feature columns")
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

print("Crossed feature columns")
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

print("Choose which columns to use")
feature_columns = []

print("Appending Numeric cols")
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

print("Appending Bucketized cols")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

print("Appending  Indicator cols")
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

print("Appending Embedding cols")
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

print("Appending Crossed cols")
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

print("Creating a feature layer")
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print("Create, compile, and train the model")
model = tf.keras.Sequential([
    feature_layer,
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

print("Configuring optimizer")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training the model")
model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

print("Evaluating the model")
loss, accuracy = model.evaluate(test_ds)

print("Accuracy: ", accuracy)
print("Program completed")
