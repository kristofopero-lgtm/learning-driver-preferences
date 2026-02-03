import matplotlib.pyplot as plt

CUSTOM_COLORS = {
    "pline_mean": "#ff0000", # red
    "pline_median":  "#ee6363", # rose
    "pline_others": "#fc9208", # orange
    "red":  "#ca0a0a", # red
    "green":  "#05C505", # green
    "grey":  "#9C9C9C", # grey
    "dark_blue": "#3C638D",   # dark blue
    "dark_orange": "#B6570F",   # dark orange
    "dark_green": "#166D56",   # dark green
    "dark_red": "#a73535",   # dark red
    }

def set_plot_style():
    plt.style.use("default")

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "lines.linewidth": 3.5,
        "lines.markersize": 6,
        "lines.marker": "o",

        "axes.prop_cycle": plt.cycler(color=[
            "#3C638D",   # dark blue
            "#B6570F",   # dark orange
            "#166D56",   # dark green
            "#a73535",   # dark red
        ])

    })
