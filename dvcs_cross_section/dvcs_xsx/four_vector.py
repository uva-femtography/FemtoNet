import tensorflow as tf


def product(a, b):
    """Four vector product
    args:
        a (tf.tensor) : tensor of shape (x, 4)
        b (tf.tensor) : tensor of shape (x, 4)
    
    returns:
        tf.tensor of rank 0
    """
    c = a * b * tf.constant([1.0, -1.0, -1.0, -1.0], dtype=tf.float32)
    axis = -1 if a.shape.rank > 1 else None
    return tf.math.reduce_sum(c, axis=axis)


def tproduct(a, b):
    """Transverse four vector product
    args:
        a (tf.tensor) : tensor of shape (x, 4)
        b (tf.tensor) : tensor of shape (x, 4)
    
    returns:
        tf.tensor of rank 0
    """
    axis = -1 if a.shape.rank > 1 else None
    return tf.math.reduce_sum(a * b, axis=axis)
