import argparse
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

from utils import attempt_switch_from_macosx, maximize_window


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json",
        required=True,
        type=str,
        help="path to history.json file of training run",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    attempt_switch_from_macosx()

    with open(args.json, "r") as f:
        logs = json.load(f)

    history = logs["history"]
    train_loss = history["loss"]
    val_loss = history["val_loss"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(train_loss)), train_loss, c="blue", label="Train")
    ax.plot(np.arange(len(val_loss)), val_loss, c="orange", label="Validation")
    ax.set_xlabel("Epoch", fontsize=13, fontname="Times")
    ax.set_ylabel("EW Huber", fontsize=13, fontname="Times")
    legend = ax.legend(fontsize=13)
    plt.setp(legend.texts, family="Times")
    plt.setp(ax.get_xticklabels(), fontsize=13, fontname="Times")
    plt.setp(ax.get_yticklabels(), fontsize=13, fontname="Times")
    plt.title(r"FemtoNet LU", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
