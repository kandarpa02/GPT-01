from GPT01.layers.attention import MultiHeadAttention
from GPT01.layers.ffn import FFN_NAMI
from tensorflow.keras.layers import Layer
import tensorflow as tf


class decoder:
    def __init__(self, dim, heads):
        self.heads = heads
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.attention = MultiHeadAttention(dim, heads)
        self.ffn = FFN_NAMI(dim, expansion=4)
    

    def call(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
        



class GPT01(Layer):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads

    def call(self, x):


