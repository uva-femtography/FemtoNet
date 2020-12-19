import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import dvcs_xsx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_path",
        required=True,
        type=str,
        help="path to csv of kinematics regions to predict",
    )
    parser.add_argument(
        "-model", required=True, type=str, help="path to saved model file"
    )
    parser.add_argument(
        "-output_path", required=True, type=str, help="path to write predictions"
    )
    parser.add_argument(
        "-p",
        "--preds",
        type=int,
        default=10,
        help="how many predictions to make while estimating uncertainty",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = pd.read_csv(args.input_path, header=None).values.astype(np.float32)

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
