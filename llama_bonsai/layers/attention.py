import tensorflow as tf
from tensorflow.keras import ops

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
    def __init__(self, dim:int, heads:int, mask=False, seed=0):
        super().__init__()
        self.qw = tf.random.normal((dim, dim), seed=seed)
        self.kw = tf.random.normal((dim, dim), seed=seed)
        self.vw = tf.random.normal((dim, dim), seed=seed)
        self.ow = tf.random.normal((dim, dim), seed=seed)
        self.heads = heads
        self.dim = dim
        self.mask = mask

    def __call__(self, x):
        Q = tf.matmul(x, self.qw)
        K = tf.matmul(x, self.kw)
        V = tf.matmul(x, self.vw)

        Q = split_heads(Q, self.heads)
        K = split_heads(K, self.heads)
        V = split_heads(V, self.heads)

        score = tf.matmul(Q, ops.swapaxes(K, -1, -2)) / ops.sqrt(self.dim)

        if self.mask:
            _, seq, _ = x.shape
            mask = ops.tril(tf.ones((1, 1, seq, seq)))
            score = score - 1e10 * (1.0-mask)

        weights = tf.nn.softmax(score, axis=-1)

        attn = tf.matmul(weights, V)
        merged = merge_heads(attn)
        out = tf.matmul(merged, self.ow)

        return out
    

# x = tf.random.uniform((3, 4, 6), seed = 0)
# print("full_size", x.shape)

# xi = split_heads(x, 2)

# print("split_size", xi.shape)

# xj = merge_heads(xi)

# print("reshape_h", xj.shape)

# print(x == xj)
