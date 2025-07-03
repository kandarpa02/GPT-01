import tensorflow as tf
from nami.TF import Nami
from tensorflow.keras import layers, Sequential


class FNN_RELU6(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.seq_model = Sequential([
            layers.Dense(units=dim * 4, activation='relu6'),
            layers.Dense(units=dim)
        ])

    def call(self, x):
        return self.seq_model(x)


class FNN_GELU(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.seq_model = Sequential([
            layers.Dense(units=dim * 4, activation='gelu'),
            layers.Dense(units=dim)
        ])

    def call(self, x):
        return self.seq_model(x)

class FNN_NAMI(tf.keras.layers.Layer):
    def __init__(self, dim):
        