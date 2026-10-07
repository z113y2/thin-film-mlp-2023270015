from pathlib import Path
import copy
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SEED = 270015
EPOCHS = 600
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_SIZES = (500, 1000, 2000, 4000)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def build_model():
    return nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 41),
        nn.Sigmoid(),
    )


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + ".png"), dpi=300)
    fig.savefig(directory / (name + ".pdf"))
    plt.close(fig)


def main():
    # Small networks often run faster with a modest CPU thread count.
    torch.set_num_threads(2)

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    models_dir = ROOT / "models"
    for directory in (results_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with np.load(ROOT / "data" / "thin_film_data.npz") as data:
        thickness = data["thickness_nm"].copy()
        spectra = data["reflectance"].copy()
        wavelengths = data["wavelengths_nm"].copy()
        train_idx = data["train_idx"].copy()
        val_idx = data["val_idx"].copy()
        test_idx = data["test_idx"].copy()
        assert int(data["seed"]) == SEED

    # Fixed physical bounds: map thickness from [40, 180] to [-1, 1].
    # This transformation does not use validation or test statistics.
    x = torch.tensor((thickness - 110.0) / 70.0, dtype=torch.float32)
    y = torch.tensor(spectra, dtype=torch.float32)
    x_val, y_val = x[val_idx], y[val_idx]
    x_test = x[test_idx]

    metrics = []
    full_prediction = None
    full_history = None

    for train_size in TRAIN_SIZES:
        # Reset initialization and shuffling seed for each experiment.
        set_seed()
        model = build_model()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )
        loss_function = nn.MSELoss()

        selected = train_idx[:train_size]
        dataset = TensorDataset(x[selected], y[selected])
        generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        history = []

        print("\nTraining with {} samples".format(train_size), flush=True)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            # Both losses are measured with the end-of-epoch model.
            model.eval()
            with torch.no_grad():
                train_loss = loss_function(
                    model(x[selected]), y[selected]
                ).item()
                val_loss = loss_function(model(x_val), y_val).item()

            history.append({
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
            })

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if epoch == 1 or epoch % 50 == 0:
                print(
                    "Epoch {:3d}/{} | train MSE {:.6f} | val MSE {:.6f}"
                    .format(epoch, EPOCHS, train_loss, val_loss),
                    flush=True,
                )

        # Select the checkpoint using validation data only.
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(x_test).numpy()

        error = prediction - spectra[test_idx]
        test_mse = float(np.mean(error ** 2))
        test_rmse = float(np.sqrt(test_mse))
        test_mae = float(np.mean(np.abs(error)))

        metrics.append({
            "train_size": train_size,
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        })

        torch.save({
            "model_state_dict": best_state,
            "train_size": train_size,
            "best_epoch": best_epoch,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "input_center_nm": 110.0,
            "input_scale_nm": 70.0,
            "architecture": "4-128-128-64-41",
            "hidden_activation": "ReLU",
            "output_activation": "Sigmoid",
        }, models_dir / "mlp_{}.pt".format(train_size))

        save_csv(
            results_dir / "history_{}.csv".format(train_size),
            history,
        )
        save_csv(results_dir / "metrics.csv", metrics)

        print(
            "Finished: best epoch {} | test RMSE {:.6f} | test MAE {:.6f}"
            .format(best_epoch, test_rmse, test_mae),
            flush=True,
        )

        if train_size == 4000:
            full_prediction = prediction
            full_history = history

    np.savez_compressed(
        results_dir / "test_predictions.npz",
        prediction=full_prediction,
        truth=spectra[test_idx],
        thickness_nm=thickness[test_idx],
        test_idx=test_idx,
        wavelengths_nm=wavelengths,
    )

    # Figure 4: losses of the model trained with 4000 samples.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(
        [row["epoch"] for row in full_history],
        [row["train_mse"] for row in full_history],
        label="Training",
    )
    ax.semilogy(
        [row["epoch"] for row in full_history],
        [row["val_mse"] for row in full_history],
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    save_figure(fig, figures_dir, "figure4_loss")

    # Figure 5: first three fixed test samples, chosen independently of error.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for position, ax in enumerate(axes):
        ax.plot(wavelengths, spectra[test_idx[position]], label="TMM")
        ax.plot(
            wavelengths, full_prediction[position],
            "--", label="MLP",
        )
        ax.set_title("Test sample {}".format(position + 1))
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_ylim(0, 1)
        ax.legend()
    save_figure(fig, figures_dir, "figure5_test_spectra")

    # Figure 6: identical test set for all training sizes.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        TRAIN_SIZES,
        [row["test_rmse"] for row in metrics],
        "o-",
    )
    ax.set_xticks(TRAIN_SIZES)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Test RMSE")
    ax.grid(alpha=0.3)
    save_figure(fig, figures_dir, "figure6_training_size")

    # Figure 8: retain the largest-error test case.
    sample_mse = np.mean(
        (full_prediction - spectra[test_idx]) ** 2, axis=1
    )
    worst_position = int(np.argmax(sample_mse))
    worst_thickness = thickness[test_idx[worst_position]]
    save_csv(results_dir / "failure_case.csv", [{
        "test_position_zero_based": worst_position,
        "dataset_index": int(test_idx[worst_position]),
        "d1_nm": worst_thickness[0],
        "d2_nm": worst_thickness[1],
        "d3_nm": worst_thickness[2],
        "d4_nm": worst_thickness[3],
        "spectrum_rmse": float(np.sqrt(sample_mse[worst_position])),
    }])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        wavelengths, spectra[test_idx[worst_position]],
        label="TMM",
    )
    ax.plot(
        wavelengths, full_prediction[worst_position],
        "--", label="MLP",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Largest-error test case: RMSE = {:.4f}"
        .format(np.sqrt(sample_mse[worst_position]))
    )
    ax.legend()
    save_figure(fig, figures_dir, "figure8_failure_case")

    print("\nAll four experiments completed.", flush=True)
    print("Metrics:", results_dir / "metrics.csv")
    print("Figures:", figures_dir)
    print("Models:", models_dir)


if __name__ == "__main__":
    main()
