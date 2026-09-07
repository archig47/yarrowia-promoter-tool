"""Regenerate every figure from results/runs.csv. Never hand-edit a figure.

D1  figures/fig1_scaling_curves.png   accuracy vs training set size, error bars,
                                      measured noise ceiling drawn on
    figures/fig2_reliability.png      seed spread vs n, and the D4a transfer ladder

Palette follows the project's existing figure convention (teal / violet), with a
warm accent reserved for the pretrained model since it carries the headline.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# measured from Native80 replicates (see STATE.md): rep1 vs rep2 Spearman
CEILING_RANDOM, CEILING_NATIVE = 0.980, 0.934

STYLE = {                      # colour, label, z-order
    "ridge":   ("#6B2D8B", "k-mer ridge", 3),
    "lgbm":    ("#9E9E9E", "LightGBM", 2),
    "cnn":     ("#0D7377", "CNN (from scratch)", 4),
    "dnabert": ("#D1495B", "DNABERT-2 (LoRA)", 5),
    "nt":      ("#E8A33D", "Nucleotide Transformer (LoRA)", 5),
}
ORDER = ["lgbm", "ridge", "cnn", "dnabert", "nt"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.dpi": 200,
})


def load():
    df = pd.read_csv(ROOT / "results" / "runs.csv")
    return df[df.cfg_split == "cluster"]          # splits agree; use one for clean error bars


def fig1(df):
    d = df[df.cfg_evalset == "primary"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    ax.axhline(CEILING_RANDOM, color="#333333", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(1.7e5, CEILING_RANDOM + 0.012, f"measured ceiling {CEILING_RANDOM:.2f}",
            va="bottom", ha="right", fontsize=7.5, color="#333333")

    for m in ORDER:
        sub = d[d.cfg_model == m]
        if not len(sub): continue
        g = sub.groupby("cfg_n").overall_spearman.agg(["mean", "std", "count"])
        c, lab, z = STYLE[m]
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=c, lw=1.6,
                    marker="o", ms=4.2, capsize=2.5, elinewidth=1, zorder=z, label=lab)

    ax.set_xscale("log")
    ax.set_xlabel("training measurements ($n$)")
    ax.set_ylabel("Spearman $\\rho$  (held-out test set)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(80, 1.9e5)
    ax.set_xticks([100, 300, 1000, 3000, 10000, 30000, 100000])
    ax.set_xticklabels(["100", "300", "1k", "3k", "10k", "30k", "100k"])
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Pretraining substitutes for measurements where data is scarce",
                 fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scaling_curves.png", bbox_inches="tight")
    fig.savefig(FIG / "fig1_scaling_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_scaling_curves.png/.pdf")


def fig2(df):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    # A: run-to-run spread vs n
    d = df[df.cfg_evalset == "primary"]
    for m in ORDER:
        sub = d[d.cfg_model == m]
        if not len(sub): continue
        g = sub.groupby("cfg_n").overall_spearman.std()
        c, lab, z = STYLE[m]
        axA.plot(g.index, g.values, color=c, lw=1.6, marker="o", ms=4.2, zorder=z, label=lab)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("training measurements ($n$)")
    axA.set_ylabel("s.d. across 3 seeds")
    axA.set_xticks([100, 300, 1000, 3000, 10000, 30000, 100000])
    axA.set_xticklabels(["100", "300", "1k", "3k", "10k", "30k", "100k"])
    axA.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0); axA.set_axisbelow(True)
    axA.legend(fontsize=7.5, loc="lower left")
    axA.set_title("A   Reliability collapses at small $n$", fontsize=9.5, loc="left")

    # B: transfer ladder at the largest n
    n_max = int(df.cfg_n.max())
    models = [m for m in ORDER if len(df[df.cfg_model == m])]
    x = np.arange(len(models)); w = 0.36
    for k, (ev, hatch, lab) in enumerate([("spikein", None, "random controls"),
                                          ("native", "///", "real yeast promoters")]):
        vals = [df[(df.cfg_model == m) & (df.cfg_evalset == ev) &
                   (df.cfg_n == n_max)].overall_spearman.mean() for m in models]
        axB.bar(x + (k - 0.5) * w, vals, w, label=lab, hatch=hatch,
                color=[STYLE[m][0] for m in models], edgecolor="white",
                alpha=1.0 if k == 0 else 0.55, linewidth=0.8)
    axB.axhline(CEILING_NATIVE, color="#333333", lw=0.9, ls=(0, (4, 3)))
    axB.text(len(models) - 0.45, CEILING_NATIVE + 0.008, f"native ceiling {CEILING_NATIVE:.2f}",
             ha="right", fontsize=7.5, color="#333333")
    axB.set_xticks(x); axB.set_xticklabels([STYLE[m][1].split(" (")[0] for m in models],
                                           fontsize=8, rotation=12, ha="right")
    axB.set_ylabel("Spearman $\\rho$")
    axB.set_ylim(0, 1.0)
    axB.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0); axB.set_axisbelow(True)
    axB.legend(fontsize=8, loc="lower left")
    axB.set_title(f"B   Real sequence costs ~0.1 $\\rho$  ($n$={n_max:,})", fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(FIG / "fig2_reliability.png", bbox_inches="tight")
    fig.savefig(FIG / "fig2_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_reliability.png/.pdf")


if __name__ == "__main__":
    df = load()
    print("models present:", sorted(df.cfg_model.unique()))
    fig1(df); fig2(df)
