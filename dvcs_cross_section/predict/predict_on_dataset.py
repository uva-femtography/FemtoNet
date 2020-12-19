import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import dvcs_xsx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", type=str, required=True, help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "-output_path", type=str, required=True, help="Path to save predictions file"
    )
    parser.add_argument(
        "--georges",
        action="store_true",
        help="Whether to load georges experiment data.",
    )
    parser.add_argument(
        "--hallB", action="store_true", help="Whether to load HallB experiment data."
    )
    parser.add_argument(
        "--defurne",
        action="store_true",
        help="Whether to load Defurne experiment data.",
    )
    parser.add_argument(
        "-p",
        "--preds",
        type=int,
        default=10,
        help="How many predictions to make while estimating uncertainty",
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

    xsx_label = dvcs_xsx.utils.parse_xsx_type(args.model)

    dataset = []

    if args.defurne:
        defurne34 = dvcs_xsx.data.load("defurne_34")
        defurne35 = dvcs_xsx.data.load("defurne_35")
        # defurne 34 has two extra columns for assymetric error
        defurne34 = defurne34[:, :-2]
        dataset += [defurne34, defurne35]

    if args.georges:
        georges_no_error = dvcs_xsx.data.load("georges_no_error")
        georges_no_error[:, -2:] = georges_no_error[:, -2:]
        georges_w_error = dvcs_xsx.data.load("georges_w_error")
        georges_w_error = georges_w_error[:, :-2]
        georges_w_error[:, -2:] = georges_w_error[:, -2:]
        dataset += [georges_w_error, georges_no_error]

    if args.hallB:
        hallB = dvcs_xsx.data.load("hallB")
        dataset += [hallB]

    dataset = np.concatenate(dataset, axis=0)
    dataset = dvcs_xsx.data.keep_observables(dataset, [xsx_label], 5)

    if dataset.shape[1] < 5:
        print("Dataset has fewer than 5 columns... Exiting")
        exit(1)

    kinematics = dataset[:, :5]

    if "sincos" in args.model:
        kinematics = dvcs_xsx.data.sin_cos_transform(kinematics)

    kinematics = tf.data.Dataset.from_tensor_slices(kinematics).batch(32)

    model = dvcs_xsx.Model().load_pretrained(args.model)

    mean_preds = []
    std_preds = []

    for batch, bin in enumerate(kinematics):
        mean, std = model.get_mean_std(bin, args.preds)
        mean_preds.append(mean)
        std_preds.append(std)

    # need to account for possibility that last batch was < batch size
    flatten = lambda x: np.concatenate([np.array(x[:-1]).flatten(), x[-1]], axis=0)

    mean_preds = np.expand_dims(flatten(mean_preds), 1)
    std_preds = np.expand_dims(flatten(std_preds), 1)

    dataset_w_preds = np.concatenate([dataset, mean_preds, std_preds], axis=1)

    np.savetxt(args.output_path, dataset_w_preds, delimiter=",")
