# SHAP-Based Explainability and Fairness Audit for Fraud Detection

**MSc Data Science and Business Analytics — PROJ519 Dissertation, University of Plymouth**

*A SHAP-Based Explainability and Fairness Audit Framework for Fraud Detection Models in UK Fintech: A Consumer Duty Compliance Perspective*

This repository contains the analysis code for the dissertation. It trains fraud detection models on the Bank Account Fraud (BAF) dataset, explains their decisions with SHAP, audits them for fairness across customer groups, and maps the findings to the FCA Consumer Duty.

---

## Contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Setting up (first time only)](#3-setting-up-first-time-only)
4. [Running the analysis](#4-running-the-analysis)
5. [Daily work routine](#5-daily-work-routine)
6. [Running on other machines](#6-running-on-other-machines)
7. [Key methodological decisions](#7-key-methodological-decisions)
8. [Results so far](#8-results-so-far)
9. [Troubleshooting](#9-troubleshooting)
10. [References](#10-references)

---

## 1. Project overview

**Aim:** to develop and evaluate a combined explainability and fairness audit framework for machine-learning fraud detection models, and to assess whether it can support evidence of compliance with the FCA Consumer Duty.

Each notebook covers one part of the proposal's objectives:

| Notebook | What it does | Proposal objective |
|---|---|---|
| `1_EDA.ipynb` | Loads and cleans the data, exploratory analysis, train/test split, first logistic regression baseline | Data understanding |
| `2_Modelling.ipynb` | Tunes and compares logistic regression (interpretable baseline) and LightGBM (gradient-boosted ensemble) | Objective 2 |
| `3_SHAP.ipynb` | Global and local SHAP explanations, cross-check against logistic regression, protected-attribute contributions | Objective 3 |
| `4_Fairness.ipynb` | Fairness audit with Fairlearn across age, income and employment status: error rates by group, summary fairness metrics, threshold sweep, drift by month, SHAP decomposition of the age gap, and retraining without protected attributes | Objectives 4 and 5 |
| *(planned)* | Validation on the IEEE-CIS Fraud Detection dataset | Generalisability check |

**Dataset:** Bank Account Fraud (BAF) Suite, Base variant (Jesus et al., 2022): 1,000,000 synthetic bank account applications over 8 months, about 1.1% fraud, with three protected attributes (age, income, employment status).

---

## 2. Repository structure

```
analysis/
├── 1_EDA.ipynb
├── 2_Modelling.ipynb
├── 3_SHAP.ipynb
├── 4_Fairness.ipynb
├── environment.yml            # list of Python packages needed
├── environment.lock.yml       # exact package versions used (if present)
├── README.md                  # this file
├── .gitignore                 # stops data and large files going to GitHub
├── data/                      # NOT on GitHub — add the dataset yourself (see below)
│   ├── Base.csv
│   └── Base.parquet           # created automatically on the first run
└── outputs/                   # created automatically by the notebooks
    ├── results/               # tables (CSV) — on GitHub
    ├── figures/               # charts (PNG) for the dissertation — on GitHub
    ├── processed/             # train/test sets, predictions, SHAP values — NOT on GitHub
    └── models/                # trained models and thresholds — NOT on GitHub
```

The data and the large files in `outputs/processed/` and `outputs/models/` are excluded by `.gitignore`. They are too large for GitHub and can be recreated by running the notebooks.

---

## 3. Setting up (first time only)

### 3.1 Install the tools

- **Git:** already installed on most Macs (type `git --version` in Terminal to check). On Windows, install it from [git-scm.com](https://git-scm.com).
- **Miniforge** (a lightweight Python and conda installer): download from [github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge). On an Apple Silicon Mac (M1/M2/M3), choose the **arm64** version.

### 3.2 Download the repository

Open Terminal (Mac) or Miniforge Prompt (Windows) and run:

```bash
cd ~/Desktop
git clone https://github.com/ghimireasmit/PROJ519-MSc-Dissertation-and-Research-Skills.git
cd PROJ519-MSc-Dissertation-and-Research-Skills/analysis
```

### 3.3 Add the dataset

The dataset is not stored on GitHub.

1. Download the BAF dataset suite from Kaggle: [Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022). Check the licence on that page before reusing or redistributing the data.
2. Create a folder called `data` inside `analysis`, and put `Base.csv` in it:

```bash
mkdir data
```

Then copy `Base.csv` into it (in Finder or File Explorer, or with `cp`).

Check it worked: `ls data` should show `Base.csv`, and `git status` should still say "working tree clean" (meaning `.gitignore` is keeping the data off GitHub).

### 3.4 Create the Python environment

From inside the `analysis` folder:

```bash
conda env create -f environment.yml
```

This takes 5–10 minutes and installs everything needed (pandas, scikit-learn, LightGBM, XGBoost, SHAP, Fairlearn, JupyterLab and others) into a separate environment called `dissertation`. You only do this once per computer.

> **Important:** run this from inside the `analysis` folder. Running it from anywhere else gives `EnvironmentFileNotFound`.

---

## 4. Running the analysis

### 4.1 Start Jupyter

```bash
conda activate dissertation
jupyter lab
```

Your Terminal prompt should change from `(base)` to `(dissertation)`. JupyterLab opens in your web browser.

### 4.2 Run the notebooks in order

Each notebook saves its results to `outputs/`, and the next notebook loads them, so **run them in order the first time**:

| Order | Notebook | Approx. time (MacBook Air M1, 8GB) |
|---|---|---|
| 1 | `1_EDA.ipynb` | 2–3 min (first run converts the CSV to Parquet) |
| 2 | `2_Modelling.ipynb` | about 5 min (includes hyperparameter tuning) |
| 3 | `3_SHAP.ipynb` | a few minutes |
| 4 | `4_Fairness.ipynb` | a few minutes (retrains both models without protected attributes) |

To run a notebook: open it, then choose **Kernel → Restart Kernel and Run All Cells…** and click **Restart**.

After the first full run, you only need to rerun a notebook when something it depends on changes. For example, notebooks 3 and 4 can be rerun on their own because they only load what notebooks 1 and 2 saved. If you change notebook 1 or 2 in a way that affects the data or the models, rerun everything after it.

### 4.3 Finishing

Save your notebooks (**Cmd + S** / **Ctrl + S**), then stop Jupyter by pressing **Control + C** twice in the Terminal window that is running it.

---

## 5. Daily work routine

Every work session:

```bash
# 1. Go to the project folder
cd "/path/to/PROJ519 MSc Dissertation and Research Skills/analysis"

# 2. Get any changes made on GitHub or another computer
git pull

# 3. Start the environment and Jupyter
conda activate dissertation
jupyter lab
```

When you've finished, save the notebooks, then in a **new** Terminal tab (Cmd + T):

```bash
cd "/path/to/PROJ519 MSc Dissertation and Research Skills/analysis"
git status                       # check what changed
git add <the files you changed>  # e.g. git add 3_SHAP.ipynb outputs/results/ outputs/figures/
git commit -m "Short description of what you did"
git push
```

**Tips**
- Tip: drag a folder from Finder into Terminal after typing `cd ` to paste its path. Paths containing spaces must be wrapped in quotes.
- Always `git pull` before starting work. This prevents "push rejected" errors.
- Commit notebooks **with their outputs**, so results and charts are visible on GitHub.

---

## 6. Running on other machines

The notebooks detect where they are running and find the data automatically, so the same notebook works unchanged in all three places.

### University PC (64GB RAM)

Same steps as section 3. On Windows, use **Miniforge Prompt** instead of Terminal. Clone the repo, add `data/Base.csv`, create the environment, then run the notebooks in order. The trained models are not on GitHub, so all notebooks must be run on each new machine.

The extra memory makes it practical to run more tuning trials (increase `N_TRIALS` in notebook 2) and the IEEE-CIS validation dataset.

> University PCs may delete local files when you log out. Keep the repo on your network drive or OneDrive.

### Google Colab

1. Upload `Base.csv` to Google Drive in a folder such as `MyDrive/Dissertation/data/`.
2. In Colab, open the notebook from GitHub (**File → Open notebook → GitHub** tab).
3. If needed, edit `COLAB_DATA_DIR` in the first code cell to match your Drive folder.
4. Install any missing packages in a new first cell, e.g. `!pip install fairlearn`.

Outputs are saved to `MyDrive/Dissertation/outputs/`.

### Pointing to the data manually

If the data is somewhere else, set the environment variable `DISSERTATION_DATA_DIR` to the folder containing `Base.csv` before starting Jupyter.

---

## 7. Key methodological decisions

These choices are made in the code and should be justified in the methodology chapter.

| Decision | What was done | Why |
|---|---|---|
| **Temporal split** | Train on months 0–4, validate on month 5, test on months 6–7. `month` is not used as a feature. | Matches the BAF paper's protocol and mimics deployment: train on the past, predict the future. |
| **Validation month** | Month 5 is used for tuning, early stopping and choosing thresholds. Months 6–7 are never used for any decision. | Keeps the test results unbiased. |
| **Decision threshold** | Set to flag 5% of legitimate applicants in month 5 (5% false-positive rate). | The standard BAF operating point. Applying it to months 6–7 shows how stable it is over time. |
| **Main metrics** | AUC-PR and recall at 5% FPR, alongside precision, recall and F1. Not accuracy. | With about 1% fraud, accuracy and AUC-ROC are misleadingly high. |
| **Categorical features** | Kept as the `category` type. One-hot encoded for logistic regression; handled natively by LightGBM. No LabelEncoder. | Keeps real category names readable for SHAP and fairness analysis, and avoids imposing a false order on categories. |
| **Hidden missing values** | In six BAF columns, negative values mean "missing". They are counted in EDA and left as they are. | Tree models can use "missing" as a signal. Documented as a modelling choice. |
| **Protected attributes** | Age, income and employment status are currently **kept as model features**. | Allows their influence to be measured with SHAP. Whether to exclude them is a decision to discuss with the supervisor. |
| **Class imbalance** | Class weighting tested as a tuning option for both models. | Neither model improved with weighting at the fixed 5% FPR operating point. |
| **Fair comparison** | Both models are trained on the same months, tuned on the same validation month and use the same threshold rule. | So any performance difference reflects the models, not the setup. |
| **SHAP method** | Exact TreeSHAP for LightGBM (via LightGBM's built-in `pred_contrib`); exact linear SHAP for logistic regression. | Both models are explained on the same log-odds scale, for every test application. |
| **Fairness operating point** | Fairness is audited at each model's deployed threshold (5% FPR on month 5), with a sweep from 1% to 10% FPR as a robustness check. | Measures the harm customers would actually experience, and shows whether disparities depend on the threshold. |
| **Main fairness measure** | False-positive rate by group (predictive equality), alongside demographic parity, disparate impact and equalised odds. | A false positive is a genuine customer wrongly declined, which is the clearest form of harm under Consumer Duty. It is also the BAF benchmark's own fairness measure. |
| **Age groups** | Decades (BAF rounds age to the decade) and a binary 50+ vs under-50 split. | 50+ is the protected group used in the BAF literature. Decades show the full pattern. |
| **Small groups** | Groups with fewer than 1,000 applicants are reported but excluded from summary metrics. 95% Wilson confidence intervals are shown for every group rate. | Prevents tiny groups (e.g. 16 applicants in their 90s) from dominating the summary metrics. |
| **Proxy analysis** | The age score gap is split into feature contributions with SHAP, and both models are retrained without the protected attributes. | Two independent ways of testing whether other features act as proxies for age. |
| **Reproducibility** | Random seed 42 throughout; package versions in `environment.yml` / `environment.lock.yml`. | So results can be reproduced exactly. |

---

## 8. Results so far

All results are on the test set: months 6–7, 205,011 applications, 2,878 frauds (1.40%).

### Model performance (notebook 2)

| Model | AUC-ROC | AUC-PR | Recall at 5% FPR | Test FPR at month-5 threshold | Genuine applicants flagged per fraud caught |
|---|---|---|---|---|---|
| Logistic regression (tuned) | 0.875 | 0.157 | 0.496 | 4.97% | 7.1 |
| LightGBM (tuned) | 0.894 | 0.200 | 0.548 | 6.03% | 7.3 |

- LightGBM catches about 5 percentage points more fraud than logistic regression at the same 5% false-positive rate, a modest gain that has to be weighed against its lower transparency.
- Logistic regression's threshold stayed on target in the test months, while LightGBM's drifted to flag 6% of genuine applicants.
- Class weighting did not improve either model at the 5% FPR operating point.

### Explainability (notebook 3)

- No single feature dominates LightGBM: the top feature (`device_os`) accounts for under 9% of total SHAP importance. The main drivers are device, phone, housing status, address history, session and email signals.
- LightGBM captures non-linear effects that logistic regression cannot (e.g. an inverted-U effect of time at current address), which helps explain its better performance.
- The two models largely agree on what matters (Spearman ρ = 0.67; 8 of their top 10 features shared; same direction for 18 of 24 numeric features). The disagreements are explained by BAF's missing-value coding, correlated features and non-linearity.
- Protected attributes account for about 11% of LightGBM's decision-making (13% for logistic regression). Among genuine applicants, age raises scores steadily from the 50s onwards in both models.
- Local explanations for a correctly flagged fraud and a wrongly flagged genuine customer were almost identical: SHAP explains *why* a score is high, but not whether the decision is right.

### Fairness (notebook 4)

| Model | FPR under 50 | FPR 50+ | Age FPR ratio | Age FPR ratio without protected attributes | Recall at 5% FPR without protected attributes |
|---|---|---|---|---|---|
| Logistic regression | 3.5% | 12.5% | 0.28 | 0.41 | 0.482 |
| LightGBM | 4.5% | 14.2% | 0.31 | 0.45 | 0.539 |

- Genuine applicants aged 50+ are about 3–3.5 times more likely to be wrongly declined. They are 16% of applicants but 38–41% of wrongly flagged genuine customers. False-positive rates also rise with income and differ sharply by employment status.
- The disparities mirror real differences in fraud rates between groups, while precision is similar across age groups: an example of fairness definitions conflicting when base rates differ.
- The age disparity persists at every threshold from 1% to 10% FPR, and becomes relatively worse at stricter thresholds. It is similar for both models, so it is not caused by model complexity.
- LightGBM's under-50 false-positive rate stayed below 5% in both test months. Its overall rate above target is driven by older applicants and by a shift in the age mix between months.
- About half of the age score gap comes from `customer_age` itself and about half from other features acting as proxies. Retraining without protected attributes closes roughly half the gap, confirming this, at a cost of about 1 percentage point of recall. Removing protected attributes reduces disparities but does not remove them.

Full tables are in `outputs/results/` and charts in `outputs/figures/`.

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `EnvironmentFileNotFound: environment.yml` | Terminal is in the wrong folder. | `cd` into `analysis` first. Check with `ls`. |
| `EnvironmentNameNotFound: dissertation` | The environment hasn't been created on this computer yet. | Run `conda env create -f environment.yml` from inside `analysis`. |
| Prompt still says `(base)` | Environment not activated. | Run `conda activate dissertation`. |
| `FileNotFoundError: Could not find data/Base.csv` | Data missing or in the wrong place. | Put `Base.csv` in `analysis/data/`. |
| `Processed data not found. Run 1_EDA.ipynb first.` or `Missing [...]. Run notebooks 2 and 3 first.` | Notebooks run out of order, or on a new machine. | Run the notebooks in order: 1, 2, 3, then 4. |
| `! [rejected] main -> main (fetch first)` when pushing | GitHub has changes your computer doesn't (e.g. an edit made on the website). | Run `git pull --rebase`, then `git push`. |
| `TypeError: string dtypes are not allowed` | Old code using `select_dtypes(include='str')` with an older pandas. | Use `include=['object', 'string']` (already done in these notebooks). |
| Laptop becomes very slow | Running low on memory (8GB). | Close other apps and browser tabs. Check **Activity Monitor → Memory**. |
| `git status` lists `data/` or `.parquet` files | `.gitignore` missing or renamed. | Make sure `.gitignore` exists in `analysis` (`ls -a` shows hidden files). **Don't commit the data**: GitHub rejects files over 100 MB. |

---

## 10. References

- Chouldechova, A. (2017) 'Fair prediction with disparate impact: A study of bias in recidivism prediction instruments', *Big Data*, 5(2), pp. 153–163.
- Financial Conduct Authority (2022) *A new Consumer Duty: Feedback to CP21/36 and final rules* (PS22/9). London: FCA.
- Jesus, S. et al. (2022) 'Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation', *NeurIPS 2022 Datasets and Benchmarks Track*. arXiv:2211.13358.
- Kleinberg, J., Mullainathan, S. and Raghavan, M. (2017) 'Inherent trade-offs in the fair determination of risk scores', *8th Innovations in Theoretical Computer Science Conference (ITCS 2017)*.
- Kozodoi, N., Jacob, J. and Lessmann, S. (2022) 'Fairness in credit scoring: Assessment, implementation and profit implications', *European Journal of Operational Research*, 297(3), pp. 1083–1094.
- Lundberg, S. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', *NeurIPS 2017*. arXiv:1705.07874.
- Lundberg, S. et al. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2, pp. 56–67.
- Slack, D. et al. (2020) 'Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods'. arXiv:1911.02508.
- Weerts, H. et al. (2023) 'Fairlearn: Assessing and Improving Fairness of AI Systems', *Journal of Machine Learning Research*, 24(257), pp. 1–8.
