# fig_utils.py
import matplotlib as mpl

def set_journal_style():
    mpl.rcParams.update({
        "figure.figsize": (5.7, 3.7),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.borderaxespad": 0.8,
        "lines.linewidth": 2.4,
        "axes.grid": False,
    })
