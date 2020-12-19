import argparse
import math

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
    parser.add_argument("-xi_low", type=float, default=0.1)
    parser.add_argument("-xi_high", type=float, default=0.5)
    parser.add_argument("-t", type=float, default=-0.172)
    parser.add_argument("-Q2", type=float, default=1.820)
    parser.add_argument("-phi", type=float, default=0.0)
    parser.add_argument("-k0", type=float, default=5.75)
    parser.add_argument(
        "-preds",
        help="how many predictions to use when estimating uncertainty",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    attempt_switch_from_macosx()

    if "UU" in args.model:
        sigma = "UU"
    elif "LU" in args.model:
        sigma = "LU"
    else:
        raise dvcs_xsx.Error(
            "Sigma could not be detected from model save path...", dvcs_xsx.WARNING
        )

    xi = np.expand_dims(np.linspace(args.xi_low, args.xi_high), 1)
    xbj = 2 * xi / (1.0 + xi)
    res = xi.shape[0]
    t = np.expand_dims(np.array([args.t for i in range(res)]), 1)
    Q2 = np.expand_dims(np.array([args.Q2 for i in range(res)]), 1)
    k0 = np.expand_dims(np.array([args.k0 for i in range(res)]), 1)
    phi = np.expand_dims(np.array([args.phi for i in range(res)]), 1)

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

    obs_to_keep = dvcs_xsx.utils.parse_xsx_type(args.model)
    dset = dvcs_xsx.data.keep_observables(dset, [obs_to_keep], column=5)
    dset = dset[np.where(dset[:, 4] == args.phi)]
    dset = dset[np.where(dset[:, 1] == args.t)]
    dset = dset[np.where(dset[:, 2] == args.Q2)]
    dset = dset[np.where(dset[:, 3] == args.k0)]
    dset = dset[np.where(dset[:, 0] >= xbj[0])]
    dset = dset[np.where(dset[:, 0] <= xbj[-1])]
    sigma_true = dset[:, -2]
    sigma_error = dset[:, -1]

    # convert true data values to xi
    dset[:, 0] = dset[:, 0] / (2 - dset[:, 0])

    fig, ax = plt.subplots(figsize=(8, 7))

    plot_uncertainty(
        ax,
        xi.squeeze(1),
        model,
        model_ready_input,
        args.preds,
        "#fcecca" if obs_to_keep == 1 else "#94efff",
        "#ffda8f" if obs_to_keep == 1 else "#3874f5",
        "#fc7600" if obs_to_keep == 1 else "#000278",
    )

    if dset.shape[0] > 0:
        # existing data error bars
        ax.errorbar(
            dset[:, 0],
            sigma_true,
            sigma_error,
            color="#070175",
            label="Data",
            fmt="o",
            capsize=4,
            capthick=2,
            zorder=3,
        )

    ax.set_xlabel(r"$\xi$", fontsize=22, fontname="Times")
    if sigma == "UU":
        ylabel = r"$\sigma_{UU}$ (nb/GeV$^4$)"
    else:
        ylabel = r"$\sigma_{LU}$ (nb/GeV$^4$)"
    ax.set_ylabel(ylabel, fontsize=22, fontname="Times")
    legend = ax.legend(fontsize=22, frameon=False)
    plt.setp(legend.texts, family="Times")
    plt.setp(ax.get_xticklabels(), fontsize=22, fontname="Times")
    plt.setp(ax.get_yticklabels(), fontsize=22, fontname="Times")
    # plt.text(r'x_B: {%s}, t: {%s}, Q^2: {%s}'.format(args.xbj, args.t, args.Q2))
    title = f"{int(math.floor(args.phi/10.)) * 10}"
    plt.text(
        0.77,
        0.6,
        r"$t$ = "
        + str(args.t)
        + r" GeV$^2$"
        + "\n"
        + r"$Q^2$ = "
        + str(args.Q2)
        + r" GeV$^2$"
        + "\n"
        + r"$\epsilon =$"
        + str(args.k0)
        + r" GeV",
        fontsize=22,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontname="Times",
    )
    plt.title(r"$\varphi$ = " + title)

    plt.subplots_adjust(left=0.175, bottom=0.15)
    # maximize_window()
    plt.show()


if __name__ == "__main__":
    main()
