import matplotlib.pyplot as plt


def maximize_window():
    mng = plt.get_current_fig_manager()
    backend = plt.get_backend()
    if backend == "TkAgg":
        mng.window.state("zoomed")
    elif backend == "wxAgg":
        mng.frame.Maximize(True)
    elif (backend == "Qt4Agg") | (backend == "Qt5Agg"):
        mng.window.showMaximized()
    else:
        print(f"maximize_window() unable to cope with backend: {backend}")
        pass


def attempt_switch_from_macosx():
    if not plt.get_backend() == "MacOSX":
        return True
    backend_list = ["Qt4Agg", "Qt5Agg", "wXAgg", "TkAgg"]
    for backend in backend_list:
        try:
            plt.switch_backend(backend)
        except:
            continue
        else:
            return True
    return False


def plot_uncertainty(ax, x, model, model_input, iters, c2, c1, cmean):
    two_below, one_below, mean, one_above, two_above = model.mean_plus_minus_two(
        model_input, iters
    )
    ax.fill_between(x, two_below, two_above, color=c2, zorder=1)
    ax.fill_between(x, one_below, one_above, color=c1, zorder=2)
    ax.plot(x, mean, color=cmean, zorder=3, label="Predictions")
