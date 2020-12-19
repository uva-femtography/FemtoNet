import argparse
import json
import os
import kerastuner
from kerastuner.tuners.hyperband import Hyperband

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dvcs_xsx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("xsx", type=str, default="UU", choices=["UU", "LU"])
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
        "-output_dir",
        type=str,
        default="saves",
        help="path to save training results (weights and logs)",
    )
    parser.add_argument(
        "-name", type=str, required=True, help="Run name (for saves directory)"
    )
    parser.add_argument("-max_epochs", default=1000, type=int)
    parser.add_argument("-hyperband_iters", default=5, type=int)
    args = parser.parse_args()
    if not args.georges and not args.hallB and not args.defurne:
        raise dvcs_xsx.Error(
            "Must load at least 1 dataset, (`--georges`, `--hallB`, `--defurne`)",
            dvcs_xsx.CRITICAL,
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    xsx_label = dvcs_xsx.utils.parse_xsx_type(args.xsx)
    dataset = []

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

    dataset = np.concatenate(dataset, axis=0)
    dataset = dvcs_xsx.data.keep_observables(dataset, [xsx_label], 5)
    train_bins, val_bins, test_bins = dvcs_xsx.data.split_by_kinematic_bins(
        dataset, train_split=0.7, val_split=0.2
    )

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

    # make a unique output dir by adding an integer to the end of the dirname
    output_dir = os.path.join(args.output_dir, f"{args.name}_{args.xsx}_{data_method}")
    i = 0
    while os.path.exists(output_dir + f"_{i}"):
        i += 1
    output_dir += f"_{i}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_dir, ckpt_dir = dvcs_xsx.utils.make_results_dirs(output_dir)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    def build_model(hp):
        activation = hp.Choice("activation", ["relu", "tanh", "swish"])
        hidden_units = []
        for hidden_layer_num in range(
            hp.Choice("hidden_layer_count", [x for x in range(1, 5)])
        ):
            hidden_units.append(
                hp.Choice(
                    f"units_layer_{hidden_layer_num}", [32, 128, 256, 400, 800, 1024],
                )
            )

        model = dvcs_xsx.Model().build_new(
            x_train.shape[1],
            hidden_units,
            activation,
            dropout_p=hp.Choice("dropout_p", [0.25, 0.35, 0.5]),
            bh_multiply=False,
        )
        normalize = hp.Choice("normalize", [True, False])
        if normalize:
            model.set_normalization(x_train)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                lr=hp.Choice("learning_rate", [0.01, 0.001, 1e-4]),
            ),
            loss=dvcs_xsx.utils.error_weighted_huber(1.0, use_ewh=True,),
            metrics=[
                dvcs_xsx.utils.median_absolute_percentage_error,
                dvcs_xsx.utils.mean_squared_error,
            ],
        )
        return model._model

    tuner = Hyperband(
        build_model,
        objective=kerastuner.Objective(
            "val_median_absolute_percentage_error", direction="min"
        ),
        max_epochs=args.max_epochs,
        max_model_size=5_000_000,
        hyperband_iterations=args.hyperband_iters,
        directory=output_dir,
        project_name=f"paramsearch_{args.name}",
    )
    tuner.search_space_summary()

    fit_args = {
        "x": x_train,
        "y": y_train,
        "validation_data": (x_val, y_val),
        "callbacks": callbacks,
        "shuffle": True,
        "batch_size": 128,
        "epochs": args.max_epochs,
    }

    history = tuner.search(**fit_args)
    model = tuner.get_best_models(num_models=1)[0]
    best_params = tuner.get_best_hyperparameters(1)[0].values
    with open(os.path.join(output_dir, f"paramsweep_final_results.json"), "w") as f:
        json.dump(best_params, f)
    model.fit(**fit_args)

    # evaluate the best model
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

    def accuracy(pred, true, error):
        assert pred.shape[0] == true.shape[0]
        true = np.squeeze(true)
        pred = np.squeeze(pred)
        upper_bound = true + error[:, 0] + error[:, 1]
        lower_bound = true - error[:, 0] - error[:, 2]
        too_high = pred >= true
        too_low = ~too_high
        correct_high = (pred <= upper_bound) * too_high
        correct_low = (pred >= lower_bound) * too_low
        total_correct = correct_high.sum() + correct_low.sum()
        return total_correct / pred.shape[0]

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    val_acc = accuracy(y_val_pred, y_val[:, 0], y_val[:, 1:])
    test_acc = accuracy(y_test_pred, y_test[:, 0], y_test[:, 1:])

    val_results["val_acc"] = val_acc
    test_results["test_acc"] = test_acc

    with open(os.path.join(log_dir, "log.json"), "w") as f:
        json.dump(test_results, f)
        json.dump(val_results, f)
        json.dump(vars(args), f)

    model.save(ckpt_dir)
