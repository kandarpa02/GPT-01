from GPT01.layers.ffn import FFN_NAMI
import tensorflow as tf

def test():
    ffn = FFN_NAMI(64)
    x = tf.random.normal((1, 32, 64))
    with tf.GradientTape() as tape:
        y = ffn(x)
    grad = tape.gradient(y, ffn.trainable_variables)
    g_act = grad[-3:]
    for i in [h.numpy() for h in g_act]:
        assert not (i == 0).all()
    assert x.shape == y.shape
