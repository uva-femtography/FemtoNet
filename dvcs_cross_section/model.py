import os

import numpy as np
import tensorflow as tf

import dvcs_xsx
from dvcs_xsx import cross_section, utils


class BH(tf.keras.layers.Layer):
    """
    Custom NN layer that compute the BH component
    of the UU cross section directly from the inputs.
    This uses the process found in the dvcs_xsx.cross_section.bh
    function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, input):
        if input.shape[1] == 5:
            xbj, t, Q2, k0, phi = tf.unstack(input, axis=-1)
        else:
            xbj, t, Q2, k0, phi, *_ = tf.unstack(input, axis=-1)
        return cross_section.bh(xbj, t, Q2, k0, phi)


class Model:
    """
        A Model makes cross section predictions based on kinematic inputs.
    """

    def __init__(self):
        self._model = None
        self._compiled = False

    def load_pretrained(self, save_path):
        """
        Load a pretrained model. This expects the tf keras SavedModel format.
        
        :param save_path: path to save folder that was created during training.
        """
        save_path = os.path.join(save_path, "saves")
        self._model = tf.keras.models.load_model(
            save_path,
            custom_objects={
                "BH": BH,
                "_error_weighted_huber": utils.error_weighted_huber(),
                "median_absolute_percentage_error": utils.median_absolute_percentage_error,
            },
        )
        return self

    def plot_model(self, *args, **kwargs):
        """
        Wrapper for tf.keras.utils.plot_model. Saves diagram of model
        architecture to disk. See Keras docs for more.
        """
        return tf.keras.utils.plot_model(self._model, *args, **kwargs)

    def summary(self, *args, **kwargs):
        """
        Wrapper for tf.keras.models.Model.summary. Prints model architecure
        summary to stdout. See Keras docs for more.
        """
        return self._model.summary(*args, **kwargs)

    def compile(self, *args, **kwargs):
        """
        Wrapper for tf.keras.models.Model.compile. Builds training graph before
        calling .fit(...). See Keras docs for more.
        """
        self._model.compile(*args, **kwargs)
        self._compiled = True

    def fit(self, *args, **kwargs):
        """
        Wrapper for tf.keras.models.Model.fit. Runs training loop. See Keras
        docs for more.
        """
        if not self._compiled:
            raise dvcs_xsx.Error("Compile model before training using .compile(...)")
        return self._model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Wrapper for tf.keraas.models.Model.evaluate. Evaluates trained model 
        on test set. See Keras docs for more.
        """
        return self._model.evaluate(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        Save model parameters to disk. See tf.keras.models.Model.save docs for more.
        """
        self._model.save(*args, **kwargs)

    def build_new(
        self,
        input_size,
        hidden_sizes,
        activation="relu",
        dropout_p=0.05,
        bh_multiply=False,
    ):
        """
        Build a new network architecture for this model.

        :param input_size: Size of input tensor (generally 5, unless using additional features)
        :param hidden_sizes: list describing the size of each hidden layer.
        :param activation: activation function string code for hidden layers.
        :param dropout_p: dropout probability. Set to 0 to remove dropout.
        """
        inp = tf.keras.layers.Input(input_size)
        main_layer_list = [tf.keras.layers.experimental.preprocessing.Normalization()]
        for size in hidden_sizes:
            main_layer_list += [
                tf.keras.layers.Dense(size),
                tf.keras.layers.Activation(activation),
                tf.keras.layers.Dropout(dropout_p),
            ]
        main_layer_list.append(tf.keras.layers.Dense(1))
        main_model = tf.keras.models.Sequential(main_layer_list)(inp)
        if bh_multiply:
            bh = BH()(inp)
            out = tf.keras.layers.Multiply()([main_model, bh])
            self._model = tf.keras.models.Model(inputs=[inp], outputs=out)
        else:
            self._model = tf.keras.models.Model(inputs=[inp], outputs=main_model)
        return self

    def set_normalization(self, x):
        # get the normalization layer from the main model
        self._model.layers[1].layers[0].adapt(x)

    def predict(self, x, training=False):
        """
        Use the model to predict the cross section at input point x.
        :param x: input tf.Tensor. Should be the correct shape for this model architecture.
        :param training: bool. Whether to enable dropout during this prediction.
        :return: tf.Tensor prediction of xsx at x.
        """
        if not self._model:
            raise dvcs_xsx.Error(
                "Attempted to use an uninitialized Model \
                    for predictions. Load a pretrained network with \
                    load_pretrained(save_path), or build a new network \
                    with build_new(...)"
            )

        return self._model(x, training=training)

    def __call__(self, x, training=False):
        return self.predict(x, training)

    def get_mean_std(self, x, iters):
        """
        Get mean and standard deviation of prediction distribution by
        making *iters* predictions on input *x*.

        :param x: input tf.Tensor.
        :param iters: int. How many times to predict on x to estimate uncertainty.
        :return: mean_pred (np.ndarray), std_pred (np.ndarray)
        """
        num_samples = x.shape[0]
        batch_preds = np.zeros((iters, num_samples))
        for pred in range(iters):
            batch_preds[pred, :] = (
                np.squeeze(self.predict(x, training=True).numpy(), 1)
                / dvcs_xsx.data.DATA_SCALE_FACTOR
            )
        mean_pred = batch_preds.mean(0)
        pred_std = batch_preds.std(0)
        return mean_pred, pred_std

    def set_weights(self, *args, **kwargs):
        return self._model.set_weight(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self._model.get_weights(*args, **kwargs)

    def mean_plus_minus_two(self, x, iters):
        """
        Convenience function that gets the mean and two standard deviations of predictions.

        :return: two_below, one_below, mean, one_above, two_above (all np.ndarrays)
        """
        mean, std = self.get_mean_std(x, iters)
        two_below = mean - 2 * std
        one_below = mean - std
        one_above = mean + std
        two_above = mean + 2 * std
        return two_below, one_below, mean, one_above, two_above
