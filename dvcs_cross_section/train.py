import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import dvcs_xsx

# cleanup numpy print outputs
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("xsx", type=str, default="UU", choices=["UU", "LU"])
    parser.add_argument(
        "-epochs",
        type=int,
        help="how many times to cycle through the dataset \
                             during training",
        default=10000,
    )
    parser.add_argument(
        "-lr", type=float, help="optimizer learning rate", default=0.001
    )
    parser.add_argument(
        "--sincos",
        help="flag to enable harmonic features in the input. \
                        generally improves performance.",
        action="store_true",
    )
    parser.add_argument(
        "--georges",
        help="flag to add Georges data to training set",
        action="store_true",
    )
    parser.add_argument(
        "--hallB",
        help="flag to add JLab Hall B data to training set",
        action="store_true",
    )
    parser.add_argument(
        "--defurne",
        help="flag to add Defurne data to training set",
        action="store_true",
    )
    parser.add_argument(
        "--pseudo", help="flag to add pseudo data to training set", action="store_true",
    )
    parser.add_argument(
        "-hidden_sizes",
        help="number and size of each hidden layer \
                        in the model. Use python list syntax, e.g. `[128, 128, 64]`",
        type=str,
        default="[128,256,256,256]",
    )
    parser.add_argument(
        "-dropout_p",
        type=float,
        help="probability that a weight \
                        is masked out during each inference step",
        default=0.25,
    )
    parser.add_argument(
        "-activation",
        type=str,
        help="activation function to use in hidden layers. \
                        Use keras string codes (e.g. `relu`, `tanh`)",
        default="relu",
    )
    parser.add_argument(
        "-delta", type=float, default=1.0, help="Huber loss delta value",
    )
    parser.add_argument(
        "--clipnorm", type=float, default=1.0, help="Gradient clipping max L2 norm",
    )
    parser.add_argument(
        "--clipvalue",
        type=float,
        default=10.0,
        help="Gradient clipping max value (elementwise)",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="saves",
        help="path to save training results (weights and logs)",
    )
    parser.add_argument(
        "--log", action="store_true",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="prevent overfitting with early stopping",
    )
    parser.add_argument(
        "-name", type=str, required=True, help="Run name (for saves directory)"
    )
    parser.add_argument(
        "--ewh", action="store_true",
    )
    parser.add_argument(
        "--mse", action="store_true",
    )
    parser.add_argument("--bh_mul", action="store_true")
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args()
    if not args.georges and not args.hallB and not args.defurne and not args.pseudo:
        raise dvcs_xsx.Error(
            "Must load at least 1 dataset, (`--georges`, `--hallB`, `--defurne`, `--pseudo`)",
            dvcs_xsx.CRITICAL,
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    xsx_label = dvcs_xsx.utils.parse_xsx_type(args.xsx)
    dataset = []

    ##################
    ## LOAD DATASET ##
    ##################

    if args.defurne:
        defurne34 = dvcs_xsx.data.load("defurne_34")
        defurne35 = dvcs_xsx.data.load("defurne_35")
        # defurne 35 is missing assymetric error
        asym_err_defurne35 = np.zeros((defurne35.shape[0], 2))
        defurne35 = np.concatenate([defurne35, asym_err_defurne35], axis=1)
        dataset += [defurne34, defurne35]

    if args.georges:
        georges_w_error = dvcs_xsx.data.load("georges_w_error")
        dataset += [georges_w_error]

    if args.hallB:
        hallB = dvcs_xsx.data.load("hallB")
        asym_err_hallB = np.zeros((hallB.shape[0], 2))
        hallB = np.concatenate([hallB, asym_err_hallB], axis=1)
        dataset += [hallB]

    if args.pseudo:
        pseudo = dvcs_xsx.data.load("unp_pseudo")
        asym_err_pseudo = np.zeros((pseudo.shape[0], 2))
        pseudo = np.concatenate([pseudo, asym_err_pseudo], axis=1)
        dataset += [pseudo]

    dataset = np.concatenate(dataset, axis=0)
    dataset = dvcs_xsx.data.keep_observables(dataset, [xsx_label], 5)
    train_bins, val_bins, test_bins = dvcs_xsx.data.split_by_kinematic_bins(
        dataset, train_split=0.7, val_split=0.2, seed=args.seed,
    )

    # split the dataset into (input, output) pairs and rescale for stability
    def _prep_dataset(bins):
        x = bins[:, :5].astype(np.float32)
        y = dvcs_xsx.data.DATA_SCALE_FACTOR * bins[:, 6:].astype(np.float32)
        return x, y

    print(f"Fitting the {args.xsx} Observable on {dataset.shape[0]} data points")
    x_train, y_train = _prep_dataset(train_bins)
    x_val, y_val = _prep_dataset(val_bins)
    x_test, y_test = _prep_dataset(test_bins)

    if args.sincos:
        x_train = dvcs_xsx.data.sin_cos_transform(x_train)
        x_val = dvcs_xsx.data.sin_cos_transform(x_val)
        x_test = dvcs_xsx.data.sin_cos_transform(x_test)
        data_method = "sincos"
    else:
        data_method = "standard"

    # build the model
    model = dvcs_xsx.Model().build_new(
        x_train.shape[1],
        eval(args.hidden_sizes),
        args.activation,
        args.dropout_p,
        bh_multiply=args.bh_mul,
    )
    model.summary()

    # make a unique output dir by adding an integer to the end of the dirname
    output_dir = os.path.join(args.output_dir, f"{args.name}_{args.xsx}_{data_method}")
    i = 0
    while os.path.exists(output_dir + f"_{i}"):
        i += 1
    output_dir += f"_{i}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_dir, ckpt_dir = dvcs_xsx.utils.make_results_dirs(output_dir)

    callbacks = []
    if args.early_stopping:
        # early stopping reduces the risk of overfitting
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            )
        )

    # set values of input normalization layer using the training set inputs
    model.set_normalization(x_train)

    ###############
    ## FIT MODEL ##
    ###############

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=args.lr, clipnorm=args.clipnorm, clipvalue=args.clipvalue
        ),
        loss="mse"
        if args.mse
        else dvcs_xsx.utils.error_weighted_huber(args.delta, args.ewh),
        metrics=[
            dvcs_xsx.utils.median_absolute_percentage_error,
            dvcs_xsx.utils.mean_squared_error,
        ],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        shuffle=True,
        batch_size=128,
    )

    ####################
    ## RECORD RESULTS ##
    ####################

    val_results = model.evaluate(x_val, y_val, batch_size=256, return_dict=True)
    val_results["val_loss"] = val_results.pop("loss")
    val_results["val_median_absolute_percentage_error"] = val_results.pop(
        "median_absolute_percentage_error"
    )

    test_results = model.evaluate(x_test, y_test, batch_size=256, return_dict=True)
    test_results["test_loss"] = test_results.pop("loss")
    test_results["test_median_absolute_percentage_error"] = test_results.pop(
        "median_absolute_percentage_error"
    )

    y_val_pred = model.predict(x_val).numpy()
    y_test_pred = model.predict(x_test).numpy()

    val_acc = dvcs_xsx.utils.accuracy(y_val_pred, y_val[:, 0], y_val[:, 1:])
    test_acc = dvcs_xsx.utils.accuracy(y_test_pred, y_test[:, 0], y_test[:, 1:])

    val_results["val_acc"] = val_acc
    test_results["test_acc"] = test_acc

    logs = {
        "history": history.history,
        "val_results": val_results,
        "test_results": test_results,
        "args": vars(args),
    }
    with open(os.path.join(log_dir, "log.json"), "w") as f:
        json.dump(logs, f)

    print(test_results)
    model.save(ckpt_dir)
