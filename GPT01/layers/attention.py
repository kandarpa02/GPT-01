import tensorflow as tf

def split_heads(vec:tf.Tensor, heads:int) -> tf.Tensor:
    batch, seq, dim = vec.shape
    h_dim = dim // heads
    vec = tf.reshape(vec, shape=[batch, seq, h_dim, heads])
    vec = tf.transpose(vec, perm=[0, 3, 1, 2])
    return vec

def merge_heads(vec:tf.Tensor) -> tf.Tensor:
    batch, heads, seq, h_dim = vec.shape
    vec = tf.transpose(vec, perm=[0, 2, 3, 1])
    vec = tf.reshape(vec, shape=[batch, seq, heads * h_dim]) # type: ignore
    return vec


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, mask=False, seed=0):
        super().__init__()
        self.qw = self.add_weight(name = "qw", shape=(dim, dim),
                                  initializer=tf.keras.initializers.RandomNormal(seed=seed))
        self.kw = self.add_weight(name = "kw", shape=(dim, dim),
                                  initializer=tf.keras.initializers.RandomNormal(seed=seed))
        self.vw = self.add_weight(name = "vw", shape=(dim, dim),
                                  initializer=tf.keras.initializers.RandomNormal(seed=seed))
        self.ow = self.add_weight(name = "ow", shape=(dim, dim),
                                  initializer=tf.keras.initializers.RandomNormal(seed=seed))
        self.heads = heads
        self.dim = dim
        self.mask = mask

    def call(self, x):
        Q = tf.matmul(x, self.qw)
        K = tf.matmul(x, self.kw)
        V = tf.matmul(x, self.vw)

        Q = split_heads(Q, self.heads)
        K = split_heads(K, self.heads)
        V = split_heads(V, self.heads)

        score = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.math.sqrt(float(self.dim))

        if self.mask:
            _, seq, _ = x.shape
            mask = tf.linalg.band_part(tf.ones((1, 1, seq, seq)), -1, 0)
            score -= 1e10 * (1.0 - mask)

        weights = tf.nn.softmax(score, axis=-1)
        attn = tf.matmul(weights, V)
        merged = merge_heads(attn)
        out = tf.matmul(merged, self.ow)
        return out
