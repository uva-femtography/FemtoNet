import io
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from termcolor import colored

import dvcs_xsx


def parse_xsx_type(model_path):
    """
    Match model name to xsx int code.
    """
    if "UU" in model_path:
        return 1
    elif "LU" in model_path:
        return 2
    else:
        raise dvcs_xsx.Error(
            f"Could not parse model type from path str: {model_path}", dvcs_xsx.Critical
        )


def error_weighted_huber(delta=1.0, use_ewh=True):
    """
    Custom loss function that uses error bar information
    to reduce the networks' sensitivity to highly inaccurate
    measurements. See the paper for more details.

    Args:
        delta (float) : Huber loss delta value (see paper)
        use_ewh (bool) : False disable the additional error
            component and recovers the standard Huber loss.
    """
    from tensorflow.python.ops import math_ops
    from tensorflow.python.framework import ops

    def huber(y_true, y_pred):
        y_pred = math_ops.cast(y_pred, dtype=tf.float32)
        y_true = math_ops.cast(y_true, dtype=tf.float32)
        error = math_ops.subtract(y_pred, y_true)
        abs_error = math_ops.abs(error)
        quadratic = math_ops.minimum(abs_error, delta)
        linear = math_ops.subtract(abs_error, quadratic)
        return math_ops.add(
            math_ops.multiply(
                ops.convert_to_tensor_v2(0.5, dtype=quadratic.dtype),
                math_ops.multiply(quadratic, quadratic),
            ),
            math_ops.multiply(delta, linear),
        )

    def _error_weighted_huber(true, pred):
        target, err, asym_up, asym_down = tf.unstack(tf.expand_dims(true, -1), axis=1)
        raw_huber = huber(target, pred)
        if use_ewh:
            err_size = 2 * err + asym_up + asym_down
            loss = raw_huber / (1.0 + abs(err_size / (target + 1e-5)))
        else:
            loss = raw_huber
        return tf.math.reduce_mean(loss)

    return _error_weighted_huber


def accuracy(pred, true, error):
    """
    Accuracy metric as described in the paper.

    Intuitively, this returns the fraction of the predictions
    that are close enough to the true values to be within
    experimental error bars. See Figure 2 in the paper for more
    information.
    """
    assert pred.shape[0] == true.shape[0]
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    upper_bound = true + error[:, 0] + error[:, 1]
    lower_bound = true - error[:, 0] - error[:, 2]

    too_high = pred >= true
    too_low = ~too_high

    # pred_err = pred - true
    correct_high = (pred <= upper_bound) * too_high
    correct_low = (pred >= lower_bound) * too_low
    total_correct = correct_high.sum() + correct_low.sum()
    return total_correct / pred.shape[0]


def mean_squared_error(true, pred):
    true = tf.expand_dims(true[:, 0], 1)
    return tf.math.reduce_mean(tf.math.square(true - pred))


def median_absolute_percentage_error(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:, 0], 1)
    ape = (abs(y_true - y_pred) / (abs(y_true) + 1e-7)) * 100.0
    median_ape = tfp.stats.percentile(ape, 50.0, interpolation="midpoint", axis=0)
    return median_ape


def make_results_dirs(save_directory):
    """
    Creates a unique place to save training information.
    """
    log_dir_path = os.path.join(save_directory, "logs")
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    ckpt_dir_path = os.path.join(save_directory, "saves")
    if not os.path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)
    return log_dir_path, ckpt_dir_path


def tensor_to_colored_string(tensor, color):
    """Print tensor's numerical value in colored ascii.

    Args:
        tensor (tf.Tensor or tf.keras.metrics.Metric) : Tensor to be printed.
        color (str) : options are grey, red, green, yellow, blue, magenta, cyan
            and white.
    
    Returns:
        None
    """
    if isinstance(tensor, tf.keras.metrics.Metric):
        val = tensor.result().numpy()
    else:
        val = tensor.numpy()
    return colored(val, color)


def tensor_to_image(tensor):
    """Convert Tensorflow matrix to standard image format (for plotting)

    Pads rank 2 Tensor with sample and channel dimensions.

    Args:
        tensor (tf.Tensor) : rank 2 Tensor.
    
    Returns:
        rank 4 tensor
    """
    return tf.expand_dims(tf.expand_dims(tensor, 0), -1)


def figure_to_image(figure):
    """Convert pyplot figure to image.

    Args:
        figure (pyplot figure)
    
    Returns:
        image of figure in a tf.Tensor.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def package_relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def optimizer_switch(opt_type):
    if opt_type == "adam":
        opt_type = tf.keras.optimizers.Adam
    elif opt_type == "rms":
        opt_type = tf.keras.optimizers.RMSprop
    elif opt_type == "sgd":
        opt_type = tf.keras.optimizers.SGD
    elif opt_type == "adadelta":
        opt_type = tf.keras.optimizers.Adadelta
    elif opt_type == "adagrad":
        opt_type = tf.keras.optimizers.Adagrad
    elif opt_type == "adamax":
        opt_type = tf.keras.optimizers.Adamax
    elif opt_type == "ftrl":
        opt_type = tf.keras.optimizers.Ftrl
    elif opt_type == "nadam":
        opt_type = tf.keras.optimizers.Nadam
    else:
        raise dvcs_xsx.Error(f"Optimizer type '{opt_type}' not recognized.")
    return opt_type


def split_on_idx(array, idx):
    return array[:idx], array[idx:]
