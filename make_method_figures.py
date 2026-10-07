from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Use a common font size and avoid Chinese text in figures to prevent
# missing-font boxes on different Windows installations.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_box(ax, x, y, w, h, text, color="#DCEAF7", fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#34526F",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#444444",
    )
    ax.add_patch(arrow)


def figure1_workflow():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.65,
        "Overall AI4S workflow",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    boxes = [
        (0.35, 3.65, 2.05, 1.05, "Random thickness\nsampling\n5000 samples"),
        (2.75, 3.65, 2.05, 1.05, "TMM calculation\nReflectance spectra\n400-800 nm"),
        (5.15, 3.65, 2.05, 1.05, "Fixed split\n4000 / 500 / 500"),
        (7.55, 3.65, 2.05, 1.05, "MLP training\n4-128-128-64-41"),
    ]

    for x, y, w, h, text in boxes:
        add_box(ax, x, y, w, h, text)

    add_arrow(ax, 2.4, 4.18, 2.7, 4.18)
    add_arrow(ax, 4.8, 4.18, 5.1, 4.18)
    add_arrow(ax, 7.2, 4.18, 7.5, 4.18)

    add_box(
        ax,
        1.5,
        1.45,
        2.2,
        1.05,
        "10000 new candidates\nMLP prediction",
        color="#E4F2DF",
    )
    add_box(
        ax,
        4.05,
        1.45,
        2.2,
        1.05,
        "Select MLP Top 10\nat 600 nm",
        color="#E4F2DF",
    )
    add_box(
        ax,
        6.6,
        1.45,
        2.2,
        1.05,
        "TMM verification\nFinal Top 5",
        color="#FCE8D5",
    )

    add_arrow(ax, 8.55, 3.6, 7.75, 2.65)
    add_arrow(ax, 3.75, 1.98, 4.0, 1.98)
    add_arrow(ax, 6.3, 1.98, 6.55, 1.98)

    ax.text(
        5,
        0.65,
        "Physics model -> data -> surrogate model -> candidate screening -> physical verification",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    save_figure(fig, "figure1_workflow")


def figure2_physical_model():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.25]},
    )

    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("Air/H/L/H/L/Glass structure", fontsize=13)

    layers = [
        ("Air", "n0 = 1.00", "#F4F4F4", 0.7),
        ("H layer 1", "nH = 2.30, d1", "#E89B86", 0.9),
        ("L layer 2", "nL = 1.45, d2", "#8DB7D9", 0.9),
        ("H layer 3", "nH = 2.30, d3", "#E89B86", 0.9),
        ("L layer 4", "nL = 1.45, d4", "#8DB7D9", 0.9),
        ("Glass substrate", "ns = 1.52", "#C8C8C8", 1.2),
    ]

    y = 6.2
    x = 1.0
    w = 3.0

    for name, info, color, height in layers:
        rect = Rectangle(
            (x, y),
            w,
            height,
            edgecolor="#333333",
            facecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + height / 2,
            name + "\n" + info,
            ha="center",
            va="center",
            fontsize=9,
        )
        y -= height

    ax.annotate(
        "Normal incidence",
        xy=(2.5, 6.25),
        xytext=(2.5, 7.0),
        ha="center",
        arrowprops=dict(arrowstyle="->", linewidth=1.3),
    )

    ax.text(
        2.5,
        0.45,
        "d1, d2, d3, d4: 40-180 nm",
        ha="center",
        fontsize=9,
    )

    ax = axes[1]
    wavelength = np.arange(400, 801, 10)
    spectrum = (
        0.28
        + 0.15 * np.sin((wavelength - 400) / 58)
        + 0.09 * np.sin((wavelength - 400) / 23)
    )
    spectrum = np.clip(spectrum, 0.03, 0.75)

    ax.plot(wavelength, spectrum, color="#2D6FA3", linewidth=2)
    ax.set_xlim(400, 800)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("TMM-generated spectrum", fontsize=13)
    ax.grid(alpha=0.25)

    ax.text(
        0.03,
        0.95,
        "5000 thickness-spectra samples\n41 wavelength points",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F4F4F4",
            edgecolor="#AAAAAA",
        ),
    )

    fig.suptitle(
        "Physical model and TMM-based data generation",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, "figure2_physical_model")


def figure3_mlp_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.65,
        "MLP surrogate model",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    groups = [
        (0.35, 2.0, 1.25, 2.2, "Input\n4 thicknesses\n[d1,d2,d3,d4]", "#DCEAF7"),
        (2.25, 1.65, 1.35, 2.9, "Hidden 1\n128\nReLU", "#E4F2DF"),
        (4.25, 1.65, 1.35, 2.9, "Hidden 2\n128\nReLU", "#E4F2DF"),
        (6.25, 1.65, 1.35, 2.9, "Hidden 3\n64\nReLU", "#E4F2DF"),
        (8.35, 1.65, 1.3, 2.9, "Output\n41 values\nSigmoid", "#FCE8D5"),
    ]

    for x, y, w, h, text, color in groups:
        add_box(ax, x, y, w, h, text, color=color, fontsize=10)

    for start, end in [(1.6, 2.2), (3.6, 4.2), (5.6, 6.2), (7.6, 8.3)]:
        add_arrow(ax, start, 3.1, end, 3.1)

    ax.text(
        5,
        0.65,
        "Input normalization: (thickness - 110) / 70     Loss: MSE     Optimizer: Adam",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    save_figure(fig, "figure3_mlp_architecture")


if __name__ == "__main__":
    figure1_workflow()
    figure2_physical_model()
    figure3_mlp_architecture()
    print("Generated Figure 1, Figure 2, and Figure 3.")
    print("Output directory:", FIG_DIR)
