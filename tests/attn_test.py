import tensorflow as tf
from llama_bonsai.layers.attention import MultiHeadAttention

def test():
    mha1 = MultiHeadAttention(64, 8)
    mha2 = MultiHeadAttention(64, 8, mask=True)
    x = tf.random.normal((8, 32, 64))
    assert x.shape == mha1(x).shape
    assert x.shape == mha2(x).shape