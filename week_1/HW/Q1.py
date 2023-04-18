import numpy as np
import tensorflow as tf

inputs = np.random.random([32, 10, 8]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(4)

output = simple_rnn(inputs)

simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = simple_rnn(inputs)
print("hi")