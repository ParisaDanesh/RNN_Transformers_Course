import numpy as np
import tensorflow as tf
from keras.datasets import mnist


# load dataset
num_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert to one-hot
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# normalize
img_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, img_size, img_size]).astype('float32') / 255
X_test = np.reshape(X_test, [-1, img_size, img_size]).astype('float32') / 255

# network params
input_shape = (img_size, img_size)
batch_size = 128
units = 256
dropout = 0.2

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(
    units=units,
    input_shape=input_shape,
))
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))
model.summary()
tf.keras.utils.plot_model(model, to_file='mnist-rnn.png', show_shapes=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'],
)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=batch_size,
)

loss, acc = model.evaluate(
    X_test, y_test,
    batch_size=batch_size
)

print("Accuracy: {:.1f}".format(100.0 * acc))
