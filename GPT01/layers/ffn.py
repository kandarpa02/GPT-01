import tensorflow as tf
from nami.TF import Nami
from tensorflow.keras import layers, Sequential


class FFN_RELU6(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.seq_model = Sequential([
            layers.Dense(units=dim * 4, activation='relu6'),
            layers.Dense(units=dim)
        ])

    def call(self, x):
        return self.seq_model(x)


class FFN_GELU(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.seq_model = Sequential([
            layers.Dense(units=dim * 4, activation='gelu'),
            layers.Dense(units=dim)
        ])

    def call(self, x):
        return self.seq_model(x)


class FFN_NAMI(tf.keras.layers.Layer):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.hidden_dim = dim * expansion
        self.fc_value = layers.Dense(self.hidden_dim)
        self.fc_gate  = layers.Dense(self.hidden_dim)
        self.proj_out = layers.Dense(dim) #                    
        self.nami = Nami()  # indigenous activation (-_-)_/

    def call(self, x):
        value = self.fc_value(x)
        gate  = self.nami(self.fc_gate(x))
        return self.proj_out(value * gate)

