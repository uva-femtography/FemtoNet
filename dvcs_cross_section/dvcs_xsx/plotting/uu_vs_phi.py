import argparse

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import numpy as np
import tensorflow as tf

import dvcs_xsx

from dvcs_xsx.plotting.utils import (
    attempt_switch_from_macosx,
    maximize_window,
    plot_uncertainty,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", required=True, type=str)
    parser.add_argument("-xbj", type=float, default=0.343)
    parser.add_argument("-t", type=float, default=-0.172)
    parser.add_argument("-Q2", type=float, default=1.820)
    parser.add_argument("-k0", type=float, default=5.75)
    parser.add_argument("--phi_min", type=int, default=0)
    parser.add_argument("--phi_max", type=int, default=360)
    parser.add_argument(
        "-preds",
        help="how many predictions to make when estimating uncertainty",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    attempt_switch_from_macosx()

    pts = args.phi_max - args.phi_min + 1
    xbj = np.expand_dims(np.array([args.xbj for i in range(pts)]), 1)
    t = np.expand_dims(np.array([args.t for i in range(pts)]), 1)
    Q2 = np.expand_dims(np.array([args.Q2 for i in range(pts)]), 1)
    k0 = np.expand_dims(np.array([args.k0 for i in range(pts)]), 1)
    phi = np.expand_dims(np.array(np.arange(args.phi_min, args.phi_max + 1) + 0.5), 1)

    dataset = []
    defurne34 = dvcs_xsx.data.load("defurne_34")
    defurne35 = dvcs_xsx.data.load("defurne_35")
    asym_err_defurne35 = np.zeros((defurne35.shape[0], 2))
    defurne35 = np.concatenate([defurne35, asym_err_defurne35], axis=1)
    dataset += [defurne34, defurne35]
    georges_w_error = dvcs_xsx.data.load("georges_w_error")
    dataset += [georges_w_error]
    hallB = dvcs_xsx.data.load("hallB")
    asym_err_hallB = np.zeros((hallB.shape[0], 2))
    hallB = np.concatenate([hallB, asym_err_hallB], axis=1)
    dataset += [hallB]
    dset = np.concatenate(dataset, axis=0).astype(np.float32)

    model = dvcs_xsx.Model().load_pretrained(args.model)

    model_ready_input = np.concatenate([xbj, t, Q2, k0, phi], axis=1)
    if "sincos" in args.model:
        model_ready_input = dvcs_xsx.data.sin_cos_transform(model_ready_input)

    dset = dvcs_xsx.data.keep_observables(dset, [1], column=5)
    dset = dset[np.where(dset[:, 0] == args.xbj)]
    dset = dset[np.where(dset[:, 1] == args.t)]
    dset = dset[np.where(dset[:, 2] == args.Q2)]
    dset = dset[np.where(dset[:, 3] == args.k0)]
    sigma_true = dset[:, -4]
    sigma_error = dset[:, -3]
    sigma_asym_down = dset[:, -2]
    sigma_asym_up = dset[:, -1]


    fig, ax = plt.subplots(figsize=(8, 7))

    plot_uncertainty(
        ax,
        phi.squeeze(1),
        model,
        model_ready_input,
        args.preds,
        "#fcecca",
        "#ffda8f",
        "#fc7600",
    )

    # existing data error bars
    lower_error_bars = np.expand_dims(sigma_error - sigma_asym_down, 1)
    upper_error_bars = np.expand_dims(sigma_error + sigma_asym_up, 1)

    ax.errorbar(
        dset[:, 4],
        sigma_true,
        np.concatenate([lower_error_bars, upper_error_bars], axis=1).T,
        color="#070175",
        label="Data",
        fmt="o",
        capsize=4,
        capthick=2,
        zorder=3,
    )

    ax.set_xlabel(r"$\varphi$", fontsize=22, fontname="Times")
    ax.set_ylabel(r"$\sigma_{UU}$ (nb/GeV$^4$)", fontsize=22, fontname="Times")
    legend = ax.legend(fontsize=22, loc="upper right")
    plt.setp(legend.texts, family="Times")
    plt.setp(ax.get_xticklabels(), fontsize=22, fontname="Times")
    plt.setp(ax.get_yticklabels(), fontsize=22, fontname="Times")

    kinematic_lbl_loc = (0.5, 0.6)
    plt.text(
        *kinematic_lbl_loc,
        r"$x_{Bj}$ = "
        + str(args.xbj)
        + "\n"
        + r"$t$ = "
        + str(args.t)
        + r" GeV$^2$"
        + "\n"
        + r"$Q^2$ = "
        + str(args.Q2)
        + r" GeV$^2$",
        fontsize=22,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontname="Times"
    )

    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.tight_layout()
    # maximize_window()
    plt.show()


if __name__ == "__main__":
    main()
