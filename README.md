# MLP-Based Spectral Prediction and Thin-Film Design

This project implements an AI4S workflow for multilayer dielectric thin films. The workflow combines the transfer matrix method (TMM) with a multilayer perceptron (MLP) surrogate model.

## Student-specific parameters

- Target wavelength: 600 nm
- Seed: 270015
- Design seed: 270016
- Structure: Air / H / L / H / L / Glass
- High-index material: nH = 2.30
- Low-index material: nL = 1.45
- Glass substrate: ns = 1.52
- Thickness range: 40–180 nm for each layer
- Wavelength range: 400–800 nm
- Wavelength interval: 10 nm
- Number of wavelength points: 41

## Dataset

A total of 5000 thickness combinations are generated using the TMM model.

The fixed data split is:

- 4000 training samples
- 500 validation samples
- 500 test samples

The input is:

```text
[d1, d2, d3, d4]
The output is the reflectance spectrum at 41 wavelength points.

MLP model
The MLP architecture is:

4-128-128-64-41
Hidden-layer activation: ReLU
Output activation: Sigmoid
Loss function: MSE
Optimizer: Adam
Learning rate: 0.001
Batch size: 64
Maximum epochs: 600
The input thickness is normalized as:

(thickness - 110) / 70
Installation
Python 3.8 or a compatible Python version is recommended.

Install the required packages with:

pip install -r requirements.txt
Reproduction steps
Run the scripts in this order:

python generate_data.py
python train_mlp.py
python screen_designs.py
The first script generates the TMM dataset.

The second script trains MLP models with 500, 1000, 2000, and 4000 training samples and generates the evaluation figures.

The third script generates 10000 new candidates, selects the top 10 candidates at 600 nm using the MLP, and verifies them using TMM.

Main output files
data/thin_film_data.npz
results/metrics.csv
results/top5_designs.csv
results/verified_top10.csv
results/failure_case.csv
results/figures/
Main results
When the training set increases from 500 to 4000 samples, the test RMSE decreases from 0.030359 to 0.011596.

For the 600 nm target wavelength, the best TMM-verified candidate among the MLP-selected top ten has:

d1 = 63.272 nm
d2 = 112.391 nm
d3 = 61.697 nm
d4 = 113.936 nm
Its reflectance at 600 nm is:

MLP prediction: 0.655636
TMM verification: 0.656041
Source files
generate_data.py
train_mlp.py
screen_designs.py
requirements.txt
README.md
