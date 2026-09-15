# Mushroom Edibility Classification — IT2011 Group Project

**Course:** IT2011 — Artificial Intelligence and Machine Learning
**Programme:** SLIIT, Year 2, Semester 1 (2026)

## Project Overview

This project builds a binary classifier that predicts whether a mushroom is **edible** or
**poisonous** based on 22 categorical physical characteristics, using the
[UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom).

- **Rows:** 8,124
- **Features:** 22 categorical columns (e.g. cap shape, odor, gill color, habitat, ...)
- **Target:** `class` — `e` (edible) / `p` (poisonous), encoded as `0` / `1`
- **Task type:** Binary classification
- **Class balance:** 4,208 edible (51.8%) vs 3,916 poisonous (48.2%) — nearly balanced
- **Most predictive feature:** `odor` — nearly perfectly separates the two classes on its own

## Group Members & Roles

> Fill in your Group ID and each member's IT number below.

| Member | IT Number | Preprocessing Technique | Model |
|---|---|---|---|
| Member 1 | IT_______ | Label Encoding | Logistic Regression |
| Member 2 | IT_______ | One-Hot Encoding | Decision Tree |
| Member 3 | IT_______ | Ordinal Encoding + Imputation Study | Random Forest |
| Member 4 | IT_______ | Binary Encoding | SVM |
| Member 5 | IT_______ | Chi-Square Feature Selection | KNN |
| Member 6 | IT_______ | Target / Frequency Encoding | Neural Network |

**Each member's preprocessing notebook is independent and starts from the raw dataset — there
is no dependency chain between member notebooks.** Every notebook in `notebooks/` loads
`../data/raw/mushrooms.csv` directly and performs its own missing-value handling and encoding
technique from start to finish, so all 6 members can work fully in parallel and each can
independently explain their own notebook end-to-end in the viva.

## Repository Structure

```
Group_ID/
├── README.md
├── requirements.txt
├── .gitignore
├── setup_git.sh
├── push_update.sh
├── data/
│   ├── raw/                                    ← mushrooms.csv lives here
│   └── external/
├── notebooks/
│   ├── IT0001_LabelEncoding.ipynb              ← Member 1
│   ├── IT0002_OneHotEncoding.ipynb             ← Member 2
│   ├── IT0003_OrdinalImputation.ipynb          ← Member 3
│   ├── IT0004_BinaryEncoding.ipynb             ← Member 4
│   ├── IT0005_ChiSquareSelection.ipynb         ← Member 5
│   └── IT0006_TargetFrequencyEncoding.ipynb    ← Member 6
├── group_pipeline.ipynb                        ← comparison notebook, NOT a dependency chain
├── models/                                     ← members add model notebooks here later
└── results/
    ├── eda_visualizations/
    ├── logs/
    └── outputs/
```

## Preprocessing Techniques at a Glance

| Notebook | Technique | Missing-value strategy | Best suited for |
|---|---|---|---|
| `IT0001_LabelEncoding.ipynb` | Label Encoding | Mode imputation | Tree-based models |
| `IT0002_OneHotEncoding.ipynb` | One-Hot Encoding | `'unknown'` category | Linear models / Neural networks |
| `IT0003_OrdinalImputation.ipynb` | Ordinal Encoding | Compares mode / `'missing'` category / row-deletion, justifies choice | Careful data-cleaning focus |
| `IT0004_BinaryEncoding.ipynb` | Binary Encoding (`category_encoders`) | Mode imputation | Balancing dimensionality |
| `IT0005_ChiSquareSelection.ipynb` | Chi-Square (χ²) Selection | Mode imputation | Reducing noise / dimensionality |
| `IT0006_TargetFrequencyEncoding.ipynb` | Target Encoding & Frequency Encoding | `'unknown'` category | Boosting models (e.g. XGBoost) |

## How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run any member notebook, in any order** — each is fully self-contained and loads
   `data/raw/mushrooms.csv` directly:
   ```bash
   jupyter notebook notebooks/IT0001_LabelEncoding.ipynb
   ```
3. **Run `group_pipeline.ipynb` last**, only after all 6 member notebooks have been run at
   least once (it reads their saved outputs from `results/outputs/` to compare them and
   produce the train/test splits for Phase 2 model training).

## Workflow Summary

**Phase 1 — Preprocessing (parallel, independent):**
Each member runs their own notebook directly against the raw CSV and pushes their notebook +
output CSV to `main` whenever done, in any order. Once all 6 outputs exist in
`results/outputs/`, anyone runs `group_pipeline.ipynb` to compare them and produce the
recommended train/test splits.

**Phase 2 — Model Training (per-member branches):**

| Branch | Member | Model |
|---|---|---|
| `model/logistic-regression` | Member 1 | Logistic Regression |
| `model/decision-tree` | Member 2 | Decision Tree |
| `model/random-forest` | Member 3 | Random Forest |
| `model/svm` | Member 4 | SVM |
| `model/knn` | Member 5 | KNN |
| `model/neural-network` | Member 6 | Neural Network |

Each member clones/pulls `main`, checks out their branch, adds their model notebook to
`models/` using the train/test split `group_pipeline.ipynb` recommended for their model, pushes,
and opens a Pull Request into `main`.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- category_encoders
- tensorflow
- nbformat
- jupyter

## Evaluation

- **Progress Review I:** Preprocessing viva — each member explains their own notebook
  (Sections 1–7) independently.
- **Final Evaluation:** Model viva + written report, covering the trained model on the
  member's assigned branch.
