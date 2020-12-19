from functools import wraps
import os
from termcolor import cprint

import tensorflow as tf

###########################
# Autograph Tracing Utils #
###########################


def trace_graphs_suggestion():
    print(
        "\n"
        "*********************************************************************************\n"
        "Running in pure eager execution mode. For better performance, trace to Tensorflow\n"
        "graphs by setting the environment variable `siwif_trace_graphs` to `True`          "
        " \n `$ export siwif_trace_graphs=True`                                           \n"
        "*********************************************************************************\n"
        "\n",
    )


try:
    TRACE_GRAPHS = (
        True if os.environ["siwif_trace_graphs"] in ["True", "true", "t"] else False
    )
except:
    TRACE_GRAPHS = False

if not TRACE_GRAPHS:
    trace_graphs_suggestion()

_TRACE_RECORD = {}


def _add_to_trace_record(func):
    global _TRACE_RECORD
    name = func.__name__
    if name in _TRACE_RECORD:
        _TRACE_RECORD[name] += 1
        count = _TRACE_RECORD[name]
        if count == 80:
            retrace_indicator(name)
    else:
        _TRACE_RECORD[name] = 1


def reset_trace_record():
    """Manually resets trace record.
    Manually clears the trace record. Used in cases where multiple
    retraces are expected (e.g. hyperparameter searches that will spawn
    many different training runs.)
    """
    global _TRACE_RECORD
    _TRACE_RECORD = {}


def trace_graph(func):
    """Drop-in replacement/extension of @tf.function decorator
    The tf.function decorator takes eager execution code and traces
    it to a computational graph that can be optimized for performance.
    However, these functions are retraced every time their call signature
    changes; there are some subtleties here that can cause
    this to occur much more often than you'd expect. This version provides
    the same functionality as tf.function but also keeps track of how many
    times each function has been traced, and prints a warning when they're
    being traced often.
    See https://www.tensorflow.org/beta/tutorials/eager/tf_function for more
    information.
    Args:
        func (function) : function to be traced by Autograph. Usually used for
            computationally expensive tensorflow ops like model predictions and loss
            functions.
    """
    if TRACE_GRAPHS:

        @wraps(func)
        @tf.function
        def func_with_trace_count(*args, **kwargs):
            _add_to_trace_record(func)
            return func(*args, **kwargs)

        return func_with_trace_count
    else:
        return func


def is_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == "while" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.while_loop.".format(f.__name__, ", ".join(map(str, args)))
        )
    elif any(node.name == "ReduceDataset" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.data.Dataset.reduce.".format(
                f.__name__, ", ".join(map(str, args))
            )
        )
    else:
        print("{}({}) gets unrolled.".format(f.__name__, ", ".join(map(str, args))))


def retrace_indicator(func_name, *args):
    print(
        f"\nWarning: {func_name} is being traced repeatedly. See "
        "https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function "
        "for more information."
    )


def retrace_args_utility(fn_name, *args):
    print(f"Retracing {fn_name}")
    for arg in args:
        print(f"{arg}")
