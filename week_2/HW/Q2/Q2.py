# Method 1
from keras.callbacks import CSVLogger
from tensorflow import keras
from keras import layers
import tensorflow as tf
import os

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "../../section2-code/sentiment-classification/aclImdb/train", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "../../section2-code/sentiment-classification/aclImdb/test", batch_size=batch_size
)


text_only_train_ds = train_ds.map(lambda x, y: x)

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

text_vectorization.adapt(text_only_train_ds)

inputs = keras.Input(shape=(1,), dtype=tf.string, name='string')
x = text_vectorization(inputs)
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(x)
x = layers.LSTM(32)(embedded)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

csv_logger = CSVLogger('q_log.csv', append=True, separator=';')

model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=[csv_logger])

model.save('./q3_method_1.keras')
