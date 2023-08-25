import os
import sys
from keras import layers
from tensorflow import keras
from keras.callbacks import CSVLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "../../section2-code/sentiment-classification/aclImdb/train", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "../../section2-code/sentiment-classification/aclImdb/test", batch_size=batch_size
)

for inputs, targets in train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break

text_only_train_ds = train_ds.map(lambda x, y: x)

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.GRU(32)(embedded)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

csv_logger = CSVLogger('./org_log.csv', append=True, separator=';')

model.fit(int_train_ds, validation_data=int_test_ds, epochs=10, callbacks=[csv_logger])

