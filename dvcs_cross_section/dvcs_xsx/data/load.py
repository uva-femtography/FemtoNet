import copy
import math
import os
import random

import numpy as np
import pandas as pd

DATA_SCALE_FACTOR = 1000.


def load_from_file(filename):
    """Use pandas to load a csv into a numpy array of 32 bit floats"""
    return pd.read_csv(filename, header=None).values.astype(np.float32)


def keep_observables(dataset, observables_to_keep, column):
    """In datasets with multiple observables (e.g. UU and LU data), we'd
    often like to keep observables of a certain type and throw away the rest.

    Args:
        dataset (np.ndarray) : input dataset numpy array
        observables_to_keep (lst(int)) : A list of observable types to keep.
                The observable type int codes are as follows:
                    1 = Unpolarized DVCS Cross Section (UU)
                    2 = Polarized DVCS Cross Section (LU)
                    3...7 = Currently Unused (N/A)
        column (int) : Rather than guess which column of your dataset contains the
                observable type, we specify its index with this integer argument.

    Returns:
        An array with the same number of columns as `dataset` but with every row
        of an observable type not given in `observables_to_keep` removed.
    """
    observables_to_remove = {1, 2, 3, 4, 5, 6, 7} - set(observables_to_keep)
    for observable in observables_to_remove:
        dataset = dataset[np.where(dataset[:, column] != observable)]
    return dataset


def load(experiment_id):
    """Primary data loading function. Reads a folder of our pre-collected
    data and creates a usable numpy array.

    Args:
        experiment_id (str) : name of the folder in FemtoNet/dvcs_cross_section/data/
            to gather data from. Examples include "georges_w_error", "defurne_34".

    Returns:
        A numpy ndarray of the dataset gathered from each of the files in the
            experiment folder.
    """
    datapath = os.path.join(os.path.dirname(__file__), experiment_id)
    dataset = None
    concat = False
    for filename in os.listdir(datapath):
        if filename[-3:] == "csv":
            full_path = os.path.join(datapath, filename)
            data = load_from_file(full_path)
            if concat == True:
                dataset = np.concatenate([dataset, data], axis=0)
            else:
                dataset = data
                concat = True
    return dataset


def flatten(bins):
    if isinstance(bins, list):
        bins = np.array(bins, dtype=object)
    if not bins.shape[0]:
        return bins
    flattened = np.expand_dims(np.zeros_like(bins[0][0]), 0)
    for group_num in range(bins.shape[0]):
        for datapoint_num in range(bins[group_num].shape[0]):
            current_bin = np.expand_dims(bins[group_num][datapoint_num], 0)
            flattened = np.concatenate([flattened, current_bin], axis=0)
    return flattened[1:]


def split_by_kinematic_bins(data, train_split=0.7, val_split=0.2, seed=231):
    """
    This function creates the train/val/test split described in our paper.

    Rather than split the total datapoints into three groups (as is common in
    ML) we split the kinematic bins into three groups, and keep all of the values
    inside those bins together. This is because the dataset is not independently
    sampled - values are collected by fixing the kinematic bins and sweeping
    through a range of \phi. Please refer to the paper for more details and
    discussion.

    Args:
        data (np.ndarray) : dataset array
        train_split (float) : float (0., 1.) determining percentage of bins placed
            in the training set.
        val_split (float) : float (0., 1.) determining percentage of bins placed
            in the validation set. (1 - train_split - val_split) is put in the test set.

    Returns:
        train_bins, val_bins, test_bins. Three numpy arrays holding the train, val
        and test set data.
    """
    data = pd.DataFrame(data).groupby([0, 1, 2])
    train_bins, val_bins, test_bins = [], [], []
    all_bin_names = copy.deepcopy(list(data.groups.keys()))
    random.seed(seed)
    random.shuffle(all_bin_names)
    train_split_num = math.floor(len(all_bin_names) * train_split)
    val_split_num = math.floor(len(all_bin_names) * val_split) + train_split_num
    for num, bin_name in enumerate(all_bin_names):
        bin = data.get_group(bin_name)
        if num <= train_split_num:
            # add to train set
            train_bins.append(bin.values)
        elif num <= val_split_num:
            # add to val set
            val_bins.append(bin.values)
        else:
            # add to test set
            test_bins.append(bin.values)
    train_bins = flatten(train_bins)
    val_bins = flatten(val_bins)
    test_bins = flatten(test_bins)
    return train_bins, val_bins, test_bins


def get_sample_count(*arrays):
    if all(array.shape[0] == arrays[0].shape[0] for array in arrays):
        sample_count = arrays[0].shape[0]
    else:
        sample_count = None
    return sample_count


def random_shuffle(*arrays):
    """Randomly shuffle variable number of np.ndarrays."""
    permutation = np.random.permutation(arrays[0].shape[0])
    shuffled = []
    for array in arrays:
        shuffled.append(array[permutation])
    return tuple(shuffled)


def sin_cos_transform(x):
    "Add 'harmonic features' of phi onto the input vector"
    phi_degrees = x[:, -1]
    phi_radians = phi_degrees * 0.0174533
    cos_phi = np.expand_dims(np.cos(phi_radians), 1)
    cos_2phi = np.expand_dims(np.cos(2 * phi_radians), 1)
    cos_3phi = np.expand_dims(np.cos(3 * phi_radians), 1)
    sin_phi = np.expand_dims(np.sin(phi_radians), 1)
    sin_2phi = np.expand_dims(np.sin(2 * phi_radians), 1)
    sin_3phi = np.expand_dims(np.sin(3 * phi_radians), 1)
    x = np.concatenate(
        [x[:, :-1], cos_phi, cos_2phi, cos_3phi, sin_phi, sin_2phi, sin_3phi], axis=1
    )
    return x


def train_val_split(*arrays, split_val=0.2):
    """Split arrays into training and validation sets. Note that this function
       splits the entire dataset. Our DLADVEP paper splits by kinematic bin
       (see split_by_kinematic_bins)

    Args:
        *arrays (np.ndarray) : variable number of numpy arrays to be split.
        split_val (float) : 0 <= split_value <= 1. What percent of the arrays
            to put in the validation set. (1-split_val is percent in training set.)
    
    Returns:
        train_set, val_set (tuple(list(np.ndarray), list(np.ndarray))) : the input arrays
        split into train and validation sets.
    """
    sample_count = get_sample_count(*arrays)
    if not sample_count:
        raise Exception(
            "Batch Axis inconsistent. All input arrays must have first axis of equal length."
        )
    arrays = random_shuffle(*arrays)
    split_idx = math.floor(sample_count * split_val)
    train_set = [array[split_idx:] for array in arrays]
    val_set = [array[:split_idx] for array in arrays]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set


if __name__ == "__main__":
    data = load("defurne_35")
    breakpoint()
