from GPT01.layers.ffn import FFN_GELU
import tensorflow as tf

def test():
    f = FFN_GELU(dim=4)
    x = tf.random.normal((1,2,4))
    assert x.shape == f(x).shape