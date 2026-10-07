from pathlib import Path
import csv

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_data import tmm_spectra, WAVELENGTHS
from train_mlp import build_model


ROOT = Path(__file__).resolve().parent
DESIGN_SEED = 270016
TARGET_NM = 600


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(270015)
    torch.use_deterministic_algorithms(True)

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(DESIGN_SEED)
    candidates = rng.uniform(40.0, 180.0, size=(10000, 4))

    # Explicitly check for exact duplicates against the original dataset.
    with np.load(ROOT / "data" / "thin_film_data.npz") as data:
        original = data["thickness_nm"].copy()
    original_rows = set(map(tuple, original))
    assert not any(tuple(row) in original_rows for row in candidates)
    assert len(set(map(tuple, candidates))) == 10000

    model = build_model()
    checkpoint = torch.load(
        ROOT / "models" / "mlp_4000.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalized = torch.tensor(
        (candidates - 110.0) / 70.0,
        dtype=torch.float32,
    )
    with torch.no_grad():
        predictions = torch.cat([
            model(batch)
            for batch in normalized.split(1024)
        ]).numpy()

    target_index = int(np.flatnonzero(WAVELENGTHS == TARGET_NM)[0])
    predicted_target = predictions[:, target_index]

    # First select exactly ten candidates using the MLP.
    top10_indices = np.argsort(
        -predicted_target, kind="stable"
    )[:10]
    top10_thickness = candidates[top10_indices]

    # Then verify those ten candidates using TMM.
    verified_spectra, transmittance = tmm_spectra(top10_thickness)
    np.testing.assert_allclose(
        verified_spectra + transmittance, 1.0,
        rtol=0.0, atol=1e-10,
    )
    verified_target = verified_spectra[:, target_index]
    verified_order = np.argsort(-verified_target, kind="stable")

    rows = []
    for rank, local_index in enumerate(verified_order, start=1):
        candidate_index = int(top10_indices[local_index])
        thickness = candidates[candidate_index]
        predicted_r = float(predicted_target[candidate_index])
        true_r = float(verified_target[local_index])

        rows.append({
            "tmm_rank_within_top10": rank,
            "mlp_rank": int(local_index) + 1,
            "candidate_index_zero_based": candidate_index,
            "d1_nm": float(thickness[0]),
            "d2_nm": float(thickness[1]),
            "d3_nm": float(thickness[2]),
            "d4_nm": float(thickness[3]),
            "mlp_R600": predicted_r,
            "tmm_R600": true_r,
            "absolute_error_R600": abs(predicted_r - true_r),
        })

    save_csv(results_dir / "verified_top10.csv", rows)
    save_csv(results_dir / "top5_designs.csv", rows[:5])

    # Preserve spectra and candidates for reproducibility.
    np.savez_compressed(
        results_dir / "design_screening.npz",
        candidates_nm=candidates,
        predicted_spectra=predictions,
        top10_indices=top10_indices,
        top10_tmm_spectra=verified_spectra,
        tmm_order_within_top10=verified_order,
        wavelengths_nm=WAVELENGTHS,
        target_nm=TARGET_NM,
        design_seed=DESIGN_SEED,
    )

    best_local_index = int(verified_order[0])
    best_candidate_index = int(top10_indices[best_local_index])
    best_tmm = verified_spectra[best_local_index]
    best_mlp = predictions[best_candidate_index]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(WAVELENGTHS, best_tmm, label="TMM verified")
    ax.plot(WAVELENGTHS, best_mlp, "--", label="MLP predicted")
    ax.axvline(
        TARGET_NM, color="gray", linestyle=":",
        label="Target: 600 nm",
    )
    ax.scatter(
        [TARGET_NM], [best_tmm[target_index]],
        color="black", zorder=5,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_ylim(0, 1)
    ax.set_title("Best TMM-verified design among MLP Top 10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "figure7_selected_design.png", dpi=300)
    fig.savefig(figures_dir / "figure7_selected_design.pdf")
    plt.close(fig)

    print("Screening completed: 10000 candidates.")
    print("Target wavelength: 600 nm")
    print("Top 10 verified using TMM; Top 5 saved.")
    print("\nRank | d1 / d2 / d3 / d4 (nm) | MLP R600 | TMM R600")
    for row in rows[:5]:
        print(
            "{} | {:.3f} / {:.3f} / {:.3f} / {:.3f} | {:.6f} | {:.6f}"
            .format(
                row["tmm_rank_within_top10"],
                row["d1_nm"], row["d2_nm"],
                row["d3_nm"], row["d4_nm"],
                row["mlp_R600"], row["tmm_R600"],
            )
        )

    print("\nSaved:", results_dir / "top5_designs.csv")
    print("Figure:", figures_dir / "figure7_selected_design.png")


if __name__ == "__main__":
    main()
