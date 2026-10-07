from pathlib import Path
import random

import numpy as np


SEED = 270015
DESIGN_SEED = 270016
TARGET_NM = 600
WAVELENGTHS = np.arange(400.0, 801.0, 10.0)
INDICES = (2.30, 1.45, 2.30, 1.45)
N_AIR = 1.0
N_GLASS = 1.52
ROOT = Path(__file__).resolve().parent


def tmm_spectra(thickness_nm):
    """Return reflectance and transmittance, shape (samples, 41)."""
    thickness_nm = np.asarray(thickness_nm, dtype=np.float64)
    thickness_nm = np.atleast_2d(thickness_nm)
    if thickness_nm.shape[1] != 4:
        raise ValueError("Each sample must contain four thicknesses.")

    shape = (len(thickness_nm), len(WAVELENGTHS))
    matrix = np.zeros(shape + (2, 2), dtype=np.complex128)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0

    # Multiply characteristic matrices from air toward the substrate.
    for layer, refractive_index in enumerate(INDICES):
        phase = (
            2.0 * np.pi * refractive_index
            * thickness_nm[:, layer, None]
            / WAVELENGTHS[None, :]
        )
        layer_matrix = np.empty_like(matrix)
        layer_matrix[..., 0, 0] = np.cos(phase)
        layer_matrix[..., 0, 1] = (
            1j * np.sin(phase) / refractive_index
        )
        layer_matrix[..., 1, 0] = (
            1j * refractive_index * np.sin(phase)
        )
        layer_matrix[..., 1, 1] = np.cos(phase)
        matrix = matrix @ layer_matrix

    b = matrix[..., 0, 0] + N_GLASS * matrix[..., 0, 1]
    c = matrix[..., 1, 0] + N_GLASS * matrix[..., 1, 1]
    denominator = N_AIR * b + c
    r = (N_AIR * b - c) / denominator
    t = 2.0 * N_AIR / denominator

    reflectance = np.abs(r) ** 2
    transmittance = (N_GLASS / N_AIR) * np.abs(t) ** 2
    return reflectance, transmittance


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Physical check: zero thickness reduces to an air/glass interface.
    bare_r, _ = tmm_spectra(np.zeros((1, 4)))
    expected_r = ((N_AIR - N_GLASS) / (N_AIR + N_GLASS)) ** 2
    np.testing.assert_allclose(bare_r, expected_r, atol=1e-12)

    thickness = rng.uniform(40.0, 180.0, size=(5000, 4))
    reflectance, transmittance = tmm_spectra(thickness)

    # No absorption: reflectance + transmittance must equal one.
    np.testing.assert_allclose(
        reflectance + transmittance, 1.0,
        rtol=0.0, atol=1e-10
    )
    assert np.isfinite(reflectance).all()
    assert np.all((reflectance >= 0.0) & (reflectance <= 1.0))

    # One fixed permutation shared by all subsequent experiments.
    permutation = rng.permutation(5000)
    train_idx = permutation[:4000]
    val_idx = permutation[4000:4500]
    test_idx = permutation[4500:]

    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "thin_film_data.npz"

    np.savez_compressed(
        output_path,
        thickness_nm=thickness,
        reflectance=reflectance,
        wavelengths_nm=WAVELENGTHS,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=SEED,
        design_seed=DESIGN_SEED,
        target_nm=TARGET_NM,
    )

    print("TMM checks passed.")
    print("Thickness shape:", thickness.shape)
    print("Spectrum shape:", reflectance.shape)
    print("Train / validation / test: 4000 / 500 / 500")
    print("Target wavelength:", TARGET_NM, "nm")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
