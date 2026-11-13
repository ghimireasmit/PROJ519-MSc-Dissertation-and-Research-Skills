# PROJ519 Dissertation Analysis

This repository contains the analysis for a dissertation studying explainability and fairness in machine-learning fraud detection using the Bank Account Fraud (BAF) dataset. The notebooks compare logistic regression and LightGBM, explain model predictions with SHAP, and audit outcomes across customer groups.

[Open the interactive notebook guide](https://ghimireasmit.github.io/PROJ519-MSc-Dissertation-and-Research-Skills/) to launch the notebooks in Google Colab one at a time. GitHub Pages hosts the guide; Python runs in Colab.

## Analysis workflow

Run the notebooks in order:

1. `1_EDA.ipynb` — explore the data, prepare train/test sets, and fit a baseline model.
2. `2_Modelling.ipynb` — tune and compare logistic regression and LightGBM.
3. `3_SHAP.ipynb` — explain model predictions with SHAP.
4. `4_Fairness.ipynb` — audit fairness across customer groups.

The later notebooks use models and processed data created by earlier notebooks.

## Data and outputs

- `data/Base.csv` is stored with Git LFS. Install Git LFS before cloning so the dataset is downloaded.
- `data/Base.parquet` is a cached copy used to speed up loading; notebook 1 can recreate it from the CSV.
- The notebooks save generated results, figures, models, and processed data under `outputs/`.

## Setup

You need Git, Git LFS, and Conda. From a terminal:

```bash
git lfs install
git clone https://github.com/ghimireasmit/PROJ519-MSc-Dissertation-and-Research-Skills.git
cd PROJ519-MSc-Dissertation-and-Research-Skills
conda env create -f environment.yml
conda activate dissertation
jupyter lab
```

Open the notebooks in JupyterLab and run them in the order listed above. The Conda environment uses Python 3.11 and is defined in `environment.yml`.

## GitHub Codespaces

Open the repository on GitHub, choose **Code → Codespaces → Create codespace on main**, and wait for setup to finish. The Codespace configuration downloads the Git LFS dataset and creates the `dissertation` Conda environment. Open `1_EDA.ipynb`, select the `dissertation` kernel, and run the notebooks in order. Codespaces usage may count toward your account's included quota or incur charges.
