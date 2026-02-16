import os
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from src.analysis.io import RENAME_MAP
from src.utils import reformat_index


def configure_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("seaborn-v0_8-paper")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.labelsize": 8,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "text.usetex": False,  # Set to True if LaTeX is available
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


configure_matplotlib_style()

COLORS = ["#2E86AB", "#F18F01", "#06A77D", "#A23B72"]  # Blue, Orange, Teal, Purple
MARKERS = ["o", "D", "^", "s"]  # Circle, Diamond, Triangle, Square
MARKER_SIZE = 6
EDGE_WIDTH = 0.8
JITTER = 0.2
TICK_FONT_SIZE = 14


def plot_model_metric_comparison(
    df: pd.DataFrame,
    bounds: tuple[pd.DataFrame, pd.DataFrame],
    metric_name: str = "Value",
    save_directory: str = None,
    grouping: str = "",
    subplot_scale: float = 0.7,
    xmax: float = None,
):
    """Plot a single metric comparison chart."""
    df = df.rename(columns=RENAME_MAP, errors="ignore")
    bounds = tuple(b.rename(columns=RENAME_MAP, errors="ignore") for b in bounds)
    groups = df.index

    fig, ax = plt.subplots(figsize=(8, len(groups) * subplot_scale))
    _plot_metric_on_axis(ax, df, *bounds, metric_name, xmax)
    _add_panel_label(ax, "topic")  # todo: hardcoded
    plt.tight_layout()

    if save_directory:
        save_path = os.path.join(
            save_directory, f"{metric_name.lower()}_model_comparison_{grouping}.png"
        )
        plt.savefig(save_path)

    return plt


def plot_model_metric_comparison_stacked(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    bounds1: tuple[pd.DataFrame, pd.DataFrame],
    bounds2: tuple[pd.DataFrame, pd.DataFrame],
    metric_name: str,
    save_directory: str = None,
    subplot_scale: float = 0.35,
    xmax: float = None,
):
    """Plot two metric comparison charts stacked vertically with shared x-axis and legend."""
    df1 = df1.rename(columns=RENAME_MAP, errors="ignore")
    df2 = df2.rename(columns=RENAME_MAP, errors="ignore")
    bounds1 = tuple(b.rename(columns=RENAME_MAP, errors="ignore") for b in bounds1)
    bounds2 = tuple(b.rename(columns=RENAME_MAP, errors="ignore") for b in bounds2)

    n_groups_1 = len(df1.index)
    n_groups_2 = len(df2.index)
    total_height = n_groups_1 + n_groups_2

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, total_height * subplot_scale),
        sharex=True,
        gridspec_kw={"height_ratios": [n_groups_1, n_groups_2], "hspace": 0.15},
    )

    _plot_metric_on_axis(
        ax1, df1, *bounds1, metric_name, xmax, show_legend=True, show_xlabel=False
    )
    _plot_metric_on_axis(
        ax2, df2, *bounds2, metric_name, xmax, show_legend=False, show_xlabel=True
    )

    _add_panel_label(ax1, "subgroup")  # todo: hardcoded
    _add_panel_label(ax2, "dimension")  # todo: hardcoded
    plt.tight_layout()

    if save_directory:
        save_path = os.path.join(
            save_directory,
            f"{metric_name.lower()}_stacked_comparison_demographics.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return plt


def _plot_metric_on_axis(
    ax: Axes,
    df: pd.DataFrame,
    lb: pd.DataFrame,
    ub: pd.DataFrame,
    metric_name: str,
    xmax: float = None,
    show_legend: bool = True,
    show_xlabel: bool = True,
):
    """Plot metric comparison on a given axis."""
    groups = df.index
    variants = df.columns

    for i, group in enumerate(groups):
        x = df.loc[group].values
        lower = lb.loc[group].values
        upper = ub.loc[group].values

        for j, (xi, variant) in enumerate(zip(x, variants)):
            y = i + (j - (len(variants) - 1) / 2) * JITTER
            xerr = [[xi - lower[j]], [upper[j] - xi]]
            ax.errorbar(
                xi,
                y,
                xerr=xerr,
                marker=MARKERS[j],
                color=COLORS[j],
                markersize=MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=EDGE_WIDTH,
                linestyle="None",
                label=variant if i == 0 else "",
                alpha=0.9,
                zorder=3,  # Ensure markers appear above grid
                elinewidth=0.3,
                capsize=3,
                capthick=0.8,
            )

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(_wrap_labels(reformat_index(groups), width=26))
    ax.margins(y=0.02)
    ax.set_ylim(-0.5, len(groups) - 0.5)

    max_x = ub.values.max() * 1.25
    ax.set_xlim(-0.005, max(max_x, xmax or max_x))
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="both", which="major", length=4, width=0.8)

    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    if show_legend:
        legend = ax.legend(
            title="Model",
            loc="best",
            frameon=True,
            fancybox=False,
            edgecolor="0.8",
            framealpha=0.95,
            title_fontsize=10,
        )
        legend.get_frame().set_linewidth(0.8)

    if show_xlabel:
        ax.set_xlabel(f"Mean {metric_name}", fontweight="normal")


def _add_panel_label(ax: Axes, label: str):
    ax.text(-0.22, 1.02, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def plot_distance_heatmap(
    distance_matrix: pd.DataFrame,
    metric_name: str = "Value",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".3f",
    save_directory: str = None,
    grouping: str = "",
):
    """
    Create a heatmap visualization for a distance matrix.

    Parameters:
    -----------
    distance_matrix : pandas.DataFrame or numpy.array
        Square matrix containing distance values between pairs
    title : str, default "Distance Matrix Heatmap"
        Title for the heatmap
    figsize : tuple, default (10, 8)
        Figure size (width, height)
    cmap : str, default 'viridis'
        Colormap for the heatmap. Good options for distance:
        'viridis', 'plasma', 'Blues', 'Reds', 'YlOrRd'
    annot : bool, default True
        Whether to annotate cells with values
    fmt : str, default '.2f'
        Format for annotations
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    if isinstance(distance_matrix, np.ndarray):
        distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.index = reformat_index(distance_matrix.index)
    distance_matrix.columns = reformat_index(distance_matrix.columns)

    min_distance, max_distance = 0, distance_matrix.max().max()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool))
    np.fill_diagonal(mask, False)

    sns.heatmap(
        distance_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=min_distance,
        vmax=max_distance,
        square=True,  # Makes cells square-shaped
        cbar_kws={"label": f"{metric_name}"},
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.tight_layout()

    if save_directory:
        save_path = os.path.join(
            save_directory, f"{metric_name.lower()}_cross_comparison_{grouping}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_boxplots(data_dict: dict, model: str, save_directory: str = None):

    labels = [
        "Conser-vative" if k == "Conservative" else k
        for k in reformat_index(data_dict.keys())
    ]
    mean_of_means = np.mean([sg[model].mean() for sg in data_dict.values()])
    print(f"Mean {model}: {mean_of_means}")
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [sg[model] for sg in data_dict.values()],
        labels=_wrap_labels(labels, width=8),
    )
    plt.ylabel("Values")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axhline(y=0)
    plt.axhline(
        y=float(mean_of_means),
        linestyle="--",
        label=f"Mean over all models: {mean_of_means:.3f}",
    )
    plt.ylim(-1, 1)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(rotation=0, fontsize=TICK_FONT_SIZE)
    plt.legend()
    if save_directory:
        save_path = os.path.join(save_directory, f"{model.lower()} boxplot.png")
        plt.savefig(save_path)


def _upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    """Return the upper-triangular (k=1) entries of a square matrix as a 1D array."""
    A = np.asarray(df)
    iu = np.triu_indices_from(A, k=1)
    return A[iu]


def _paired_upper_triangle(
    corr_wvs: pd.DataFrame, corr_model: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned upper-triangle vectors (exclude NaNs)."""
    x = _upper_triangle_values(corr_wvs).astype(float)
    y = _upper_triangle_values(corr_model).astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def _wrap_labels(labels, width: int):
    return ["\n".join(wrap(label, width)) for label in labels]
