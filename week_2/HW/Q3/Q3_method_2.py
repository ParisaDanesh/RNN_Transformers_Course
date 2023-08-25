import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the IMDB reviews dataset - raw data
imdb_reviews = tfds.load('imdb_reviews')

# split dataset into train and test
train_raw = imdb_reviews['train']
test_raw = imdb_reviews['test']


# format dataset into a tuple of (text_data, label)
def format_dataset(input_data):
    return input_data['text'], input_data['label']


train_dataset = train_raw.map(format_dataset)
test_dataset = test_raw.map(format_dataset)

text_dataset = train_raw.map(lambda data: data['text'])

# visualize some data
for item in train_raw.take(4):
    print(item['label'].numpy())
    print(item['text'].numpy())

# Preparing data
import string
import re

max_features = 20000
embedding_dim = 128

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=500)

vectorize_layer.adapt(text_dataset.batch(64))

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = tf.keras.layers.Embedding(max_features + 1, embedding_dim)(x)
x = tf.keras.layers.LSTM(32)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(text_input, outputs)
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=["accuracy"])

model.fit(train_dataset.batch(32),
          validation_data=test_dataset.batch(32),
          epochs=10)

model.save('./q3.keras')
