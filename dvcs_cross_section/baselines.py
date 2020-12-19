import sklearn

from sklearn import preprocessing, metrics, model_selection

import numpy as np
import argparse
import os
import random
import copy
import json
import pickle

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
        "-model_type", type=str, default="linear",
    )
    parser.add_argument(
        "-name", type=str, required=True, help="Run name (for saves directory)"
    )
    parser.add_argument(
        "--pseudo", help="flag to add pseudo data to training set", action="store_true",
    )
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

    if args.pseudo:
        pseudo = dvcs_xsx.data.load("unp_pseudo")
        asym_err_pseudo = np.zeros((pseudo.shape[0], 2))
        pseudo = np.concatenate([pseudo, asym_err_pseudo], axis=1)
        dataset += [pseudo]

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

    # sklearn normalization. Use train set to compute statistics,
    # then apply those statistics to the test set.
    normalizer = preprocessing.StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_val = normalizer.transform(x_val)
    x_test = normalizer.transform(x_test)

    y_train_ = y_train[:, 0]
    y_val_ = y_val[:, 0]
    y_test_ = y_test[:, 0]

    if args.model_type == "linear":
        from sklearn import linear_model

        model_type = linear_model.LinearRegression
        param_grid = {}

    elif args.model_type == "svm":
        from sklearn import svm

        model_type = svm.SVR
        param_grid = {
            "kernel": ["poly", "rbf"],
            "degree": [2, 5, 7],
            "gamma": ["scale", "auto"],
            "C": [0.5, 1.0, 3.0, 10.0],
        }
    elif args.model_type == "knn":
        from sklearn import neighbors

        model_type = neighbors.KNeighborsRegressor
        param_grid = {
            "n_neighbors": [2, 3, 5, 7, 10, 20],
            "weights": ["uniform", "distance"],
            "p": [1, 2, 4, 8],
        }

    elif args.model_type == "forest":
        from sklearn import ensemble

        model_type = ensemble.RandomForestRegressor
        param_grid = {
            "n_estimators": [50, 100, 150, 200],
            "min_samples_split": [2, 5, 8],
            "max_features": ["auto", "sqrt"],
        }

    # sklearn trickery that lets us use our own train/val split, rather than
    # the standard cross validation
    x_search = np.concatenate((x_train, x_val), axis=0)
    y_search = np.concatenate((y_train_, y_val_))
    split_idx = [-1 for i in range(len(x_train))] + [0 for j in range(len(x_val))]
    train_val_split = model_selection.PredefinedSplit(test_fold=split_idx)

    def med_ape(true, predicted):
        # reformat to use same med ape implementation as the nn
        true = np.concatenate(
            [np.expand_dims(true, 1), np.zeros((true.shape[0], 3))], axis=1
        )
        return dvcs_xsx.utils.median_absolute_percentage_error(
            true, np.expand_dims(predicted, 1)
        ).numpy()

    # grid search for best hparams
    search = model_selection.GridSearchCV(
        model_type(),
        param_grid,
        metrics.make_scorer(med_ape, greater_is_better=False),
        cv=train_val_split,
        refit=False,
    )

    search.fit(x_search, y_search)

    # fit the best model (using only the training set!)
    model = model_type(**search.best_params_)
    model.fit(x_train, y_train_)

    # calculate performance stats
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    val_mse = float(metrics.mean_squared_error(y_val_pred, y_val_))
    test_mse = float(metrics.mean_squared_error(y_test_pred, y_test_))

    val_re = abs(y_val_pred - y_val_) / (abs(y_val_) + 1e-7)
    test_re = abs(y_test_pred - y_test_) / (abs(y_test_) + 1e-7)

    val_med_ape = med_ape(y_val_, y_val_pred).item()
    test_med_ape = med_ape(y_test_, y_test_pred).item()

    val_acc = dvcs_xsx.utils.accuracy(y_val_pred, y_val[:, 0], y_val[:, 1:])
    test_acc = dvcs_xsx.utils.accuracy(y_test_pred, y_test[:, 0], y_test[:, 1:])

    results = {
        "model_type": args.model_type,
        "test_mse": test_mse,
        "val_mse": val_mse,
        "val_med_ape": val_med_ape,
        "test_med_ape": test_med_ape,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }

    with open(os.path.join(log_dir, "log.json"), "w") as f:
        json.dump(results, f)
        json.dump(vars(args), f)

    with open(os.path.join(ckpt_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(results)
