from GPT01.layers.ffn import FFN_RELU6
import tensorflow as tf

def test():
    f = FFN_RELU6(dim=4)
    x = tf.random.normal((1,2,4))
    assert x.shape == f(x).shape