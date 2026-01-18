# -*- coding: utf-8 -*-
"""
Journal-ready capacity/bounds figures for AD, GADC, and depolarizing channels.

- Color- and grayscale-safe styling (distinct colors + linestyles + markers)
- Extra curves: Reverse Coherent Information (RCI) and an SDP/max-Rains–type upper bound
- Robust labels (no fragile LaTeX commands like \\tfrac)
- Optional CSV overrides for exact data from solvers
- Small helpers to emit CSV from NPZ/HDF5 (if h5py is installed)

USAGE
-----
$ python make_plots.py

Outputs to ./figs:
  fig_ad_single.{pdf,png}
  fig_ad_gadc_panel.{pdf,png}
  fig_dep_bounds.{pdf,png}
  fig_gadc_N0.0.{pdf,png}
  fig_gadc_N0.1.{pdf,png}
  fig_gadc_N0.2.{pdf,png}

CSV OVERRIDES (optional)
------------------------
Put CSV files in ./data with two columns (no header): x,y

AD:
  ad_ic.csv, ad_rci.csv, ad_ce.csv, ad_upper.csv
GADC (for N_th in {0.0,0.1,0.2}):
  gadc_ic_N0.0.csv, gadc_rci_N0.0.csv, gadc_ce_N0.0.csv, gadc_upper_N0.0.csv
  gadc_ic_N0.1.csv, ...
  gadc_ic_N0.2.csv, ...
Depolarizing:
  dep_hashing.csv, dep_upper.csv

Flip USE_CSV_DATA below to True when your CSVs are ready.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------- CONFIG ---------------------------------------------

OUTDIR = "figs"
DATADIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# Toggle to prefer CSVs (if present)
USE_CSV_DATA = False

# Matplotlib style (journal-ish)
mpl.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 170,
})

# Color-blind friendly palette + grayscale-safe line encodings
COL_IC     = "#1f77b4"  # blue
COL_RCI    = "#2ca02c"  # green
COL_CE     = "#e69f00"  # orange
COL_UPPER  = "#444444"  # dark grey

STY_IC    = {"linestyle": "-",  "marker": "o", "linewidth": 2.6, "markersize": 6}
STY_RCI   = {"linestyle": "--", "marker": "s", "linewidth": 2.6, "markersize": 6}
STY_CE    = {"linestyle": "-.", "marker": "^", "linewidth": 2.6, "markersize": 6}
STY_UPPER = {"linestyle": ":",  "marker": "D", "linewidth": 2.6, "markersize": 6}

# ---------------------------------------------------------------------------

def read_csv_or(x_def, y_def, fname):
    """Return (x, y). If CSV exists (and USE_CSV_DATA), load it; else return defaults."""
    fpath = os.path.join(DATADIR, fname)
    if USE_CSV_DATA and os.path.exists(fpath):
        arr = np.loadtxt(fpath, delimiter=",")
        return arr[:, 0], arr[:, 1]
    return np.array(x_def, float), np.array(y_def, float)

def savefig(fig, name):
    pdf = os.path.join(OUTDIR, f"{name}.pdf")
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    print(f"saved: {pdf}\n       {png}")

# ----------------------- DEFAULT ARRAYS (illustrative placeholders) ----------

def ad_defaults():
    gamma = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.49, 0.60])
    Ic    = np.array([1.00, 0.82, 0.70, 0.50, 0.33, 0.12, 0.00, 0.00])  # LB
    RCI   = np.maximum(0, Ic - 0.08)               # a bit below Ic
    CE    = 2.05 - 1.85 * gamma                    # E-assisted cap (illustrative trend)
    UPP   = np.maximum(CE + 0.05, 1.8 - 1.5*gamma) # ensure upper ≥ CE
    return gamma, Ic, RCI, CE, UPP

def gadc_defaults(Nth=0.1):
    gamma = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60])
    base  = 1.0 - 0.9*Nth
    Ic    = np.maximum(0, base - 1.9*gamma)         # LB
    RCI   = np.maximum(0, Ic - (0.06 + 0.1*Nth))    # LB slightly below Ic
    CE    = 2.05 - 1.85*gamma                       # illustrative
    UPP   = np.maximum(CE + 0.05, 1.8 - 1.6*gamma)  # upper ≥ CE
    return gamma, Ic, RCI, CE, UPP

def depol_defaults():
    p = np.linspace(0.0, 0.25, 11)
    hashing = np.maximum(0, 1.0 - 5.3*p + 3.2*p**2)
    hashing[p >= 0.19] = 0.0
    upper = 1.6 - 3.4*p
    return p, hashing, upper

# ----------------------- PLOTTING -------------------------------------------

def plot_channel(ax, x, curves, title=None, vline_half=True):
    for y, lab, col, sty in curves:
        ax.plot(x, y, label=lab, color=col, **sty)

    if vline_half:
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=2)
        ax.text(0.505, 0.04, "γ = 1/2", color="gray", fontsize=15)

    ax.set_xlim([0.0, max(x)])
    ymax = max([float(np.nanmax(y)) for y, *_ in curves] + [1e-9]) * 1.05
    ax.set_ylim([0.0, ymax])
    ax.set_xlabel("Damping γ")
    ax.set_ylabel("Rate / bound (bits/use)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=1)
    ax.legend(frameon=True, facecolor="white", framealpha=0.85, borderpad=0.6)

def fig_ad_single():
    g_def, ic_def, rci_def, ce_def, upp_def = ad_defaults()
    g,  Ic   = read_csv_or(g_def, ic_def, "ad_ic.csv")
    _,  RCI  = read_csv_or(g_def, rci_def, "ad_rci.csv")
    _,  CE   = read_csv_or(g_def, ce_def,  "ad_ce.csv")
    _,  UPP  = read_csv_or(g_def, upp_def, "ad_upper.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    curves = [
        (Ic,   r"Coh. info (1-shot)  $I_c$", COL_IC,    STY_IC),
        (RCI,  r"Reverse coh. info (RCI)",  COL_RCI,   STY_RCI),
        (CE,   r"Ent.-assisted  $C_E$",      COL_CE,    STY_CE),
        (UPP,  r"SDP / max-Rains (upper)",  COL_UPPER, STY_UPPER),
    ]
    plot_channel(ax, g, curves, title="Amplitude damping: comparisons vs γ", vline_half=True)
    savefig(fig, "fig_ad_single")

def fig_ad_gadc_panel(Nth=0.1):
    # AD data
    g_ad_d, ic_ad_d, rci_ad_d, ce_ad_d, upp_ad_d = ad_defaults()
    g_ad, Ic_ad = read_csv_or(g_ad_d, ic_ad_d, "ad_ic.csv")
    _,    RCI_ad = read_csv_or(g_ad_d, rci_ad_d, "ad_rci.csv")
    _,    CE_ad  = read_csv_or(g_ad_d, ce_ad_d,  "ad_ce.csv")
    _,    UPP_ad = read_csv_or(g_ad_d, upp_ad_d, "ad_upper.csv")

    # GADC data
    g_g_d, ic_g_d, rci_g_d, ce_g_d, upp_g_d = gadc_defaults(Nth)
    g_g, Ic_g  = read_csv_or(g_g_d, ic_g_d,  f"gadc_ic_N{Nth}.csv")
    _,   RCI_g = read_csv_or(g_g_d, rci_g_d, f"gadc_rci_N{Nth}.csv")
    _,   CE_g  = read_csv_or(g_g_d, ce_g_d,  f"gadc_ce_N{Nth}.csv")
    _,   UPP_g = read_csv_or(g_g_d, upp_g_d, f"gadc_upper_N{Nth}.csv")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    curves_ad = [
        (Ic_ad,  "$I_c$", COL_IC,    STY_IC),
        (RCI_ad, "RCI",   COL_RCI,   STY_RCI),
        (CE_ad,  "$C_E$", COL_CE,    STY_CE),
        (UPP_ad, "upper", COL_UPPER, STY_UPPER),
    ]
    curves_g = [
        (Ic_g,  "$I_c$", COL_IC,    STY_IC),
        (RCI_g, "RCI",   COL_RCI,   STY_RCI),
        (CE_g,  "$C_E$", COL_CE,    STY_CE),
        (UPP_g, "upper", COL_UPPER, STY_UPPER),
    ]
    plot_channel(axs[0], g_ad, curves_ad, title="AD", vline_half=True)
    plot_channel(axs[1], g_g,  curves_g,  title=f"GADC (N_th = {Nth})", vline_half=True)
    axs[0].set_ylabel("Rate / bound (bits/use)")
    savefig(fig, "fig_ad_gadc_panel")

def fig_gadc_single(Nth):
    g_d, ic_d, rci_d, ce_d, upp_d = gadc_defaults(Nth)
    g, Ic  = read_csv_or(g_d, ic_d,  f"gadc_ic_N{Nth}.csv")
    _, RCI = read_csv_or(g_d, rci_d, f"gadc_rci_N{Nth}.csv")
    _, CE  = read_csv_or(g_d, ce_d,  f"gadc_ce_N{Nth}.csv")
    _, UPP = read_csv_or(g_d, upp_d, f"gadc_upper_N{Nth}.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    curves = [
        (Ic,  r"Coh. info (1-shot)  $I_c$", COL_IC,    STY_IC),
        (RCI, "RCI",                 COL_RCI,   STY_RCI),
        (CE,  r"Ent.-assisted  $C_E$", COL_CE,    STY_CE),
        (UPP, "upper",              COL_UPPER, STY_UPPER),
    ]
    plot_channel(ax, g, curves, title=f"GADC: comparisons vs γ  (N_th = {Nth})", vline_half=True)
    savefig(fig, f"fig_gadc_N{Nth}")

def fig_dep_bounds():
    p_d, hashing_d, upper_d = depol_defaults()
    p, hashing = read_csv_or(p_d, hashing_d, "dep_hashing.csv")
    _, upper   = read_csv_or(p_d, upper_d,  "dep_upper.csv")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(p, hashing, label="Hashing LB", color=COL_IC, **STY_IC)
    ax.plot(p, upper,   label="SDP / max-Rains (upper)", color=COL_UPPER, **STY_UPPER)
    ax.set_xlim([0.0, float(np.max(p))])
    ax.set_ylim([0.0, float(np.max(upper))*1.05])
    ax.set_xlabel("Depolarizing probability $p$")
    ax.set_ylabel("Rate / bound (bits/use)")
    ax.set_title("Qubit Depolarizing: Lower/Upper Bounds")
    ax.grid(alpha=0.25, linewidth=1)
    ax.legend(frameon=True, facecolor="white", framealpha=0.85, borderpad=0.6)
    savefig(fig, "fig_dep_bounds")

# ----------------------- CSV EMITTERS (optional) -----------------------------

def emit_csv_from_npz(npz_path, key_x, key_y, out_csv):
    """
    Load arrays from NPZ and write CSV with two columns x,y (no header).
    Example:
      emit_csv_from_npz("runs/out_ad_ic.npz", "gamma", "Ic", "data/ad_ic.csv")
    """
    arr = np.load(npz_path)
    x = np.asarray(arr[key_x]).ravel()
    y = np.asarray(arr[key_y]).ravel()
    xy = np.column_stack([x, y])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, xy, delimiter=",")
    print(f"[CSV] wrote {out_csv} from {npz_path}:{key_x},{key_y}")

def emit_csv_from_hdf5(h5_path, key_x, key_y, out_csv):
    """
    Same for HDF5, only if h5py is available.
    """
    try:
        import h5py
    except Exception:
        raise RuntimeError("h5py not installed; cannot read HDF5.")
    with h5py.File(h5_path, "r") as f:
        x = np.asarray(f[key_x][...]).ravel()
        y = np.asarray(f[key_y][...]).ravel()
    xy = np.column_stack([x, y])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, xy, delimiter=",")
    print(f"[CSV] wrote {out_csv} from {h5_path}:{key_x},{key_y}")

# ----------------------- MAIN ------------------------------------------------

def main():
    fig_ad_single()
    fig_ad_gadc_panel(Nth=0.1)
    fig_dep_bounds()
    for v in [0.0, 0.1, 0.2]:
        fig_gadc_single(Nth=v)
    print(f"[OK] Wrote figures to: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
