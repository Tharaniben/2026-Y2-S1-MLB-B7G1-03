# Claude Code Prompt — SLIIT IT2011 Mushroom ML Project (v2 — Independent Notebooks)

Paste this entire prompt into Claude Code. It will generate every file for the project.

---

## INSTRUCTIONS FOR CLAUDE CODE

Create a complete machine learning university project. Generate all files listed below with
fully working, well-commented Python code in Jupyter notebook (.ipynb) format using nbformat.

**IMPORTANT DESIGN RULE:** Each member's preprocessing notebook is INDEPENDENT. Every notebook
loads the RAW dataset directly (`../data/raw/mushrooms.csv`) and does its own missing-value
handling and its own encoding technique from start to finish. No notebook should load another
member's output file. This lets all 6 members work in parallel and each fully explain their own
notebook in the viva without depending on anyone else's work.

---

## PROJECT CONTEXT

- Course: IT2011 — Artificial Intelligence and Machine Learning (SLIIT, Year 2, Semester 1)
- Task: Predict whether a mushroom is edible or poisonous (binary classification)
- Dataset: UCI Mushroom Dataset (user has a CSV file with headers already added)
- Group: 6 members — each handles one INDEPENDENT preprocessing technique AND one ML model
- Evaluation: Progress Review I (preprocessing viva) + Final Evaluation (model viva + report)

---

## DATASET FACTS (critical — use these throughout all notebooks)

- File path: `../data/raw/mushrooms.csv`
- Rows: 8,124 | Features: 22 categorical columns + 1 target
- Target column: `class` — values are `'e'` (edible) and `'p'` (poisonous)
- Encode target as: `e → 0`, `p → 1`
- All 22 feature columns contain single-letter categorical codes
- **Missing values**: `stalk-root` column uses `'?'` to represent missing entries (~30% missing)
- **Useless feature**: `veil-type` has only one unique value (`'p'`) — each member should drop it
  after noting it, since it adds no information
- **Class balance**: 4208 edible (51.8%) and 3916 poisonous (48.2%) — nearly balanced
- **Most predictive feature**: `odor` — nearly perfectly separates classes on its own

Column names in order (comma-separated, 23 total):
```
class, cap-shape, cap-surface, cap-color, bruises, odor, gill-attachment,
gill-spacing, gill-size, gill-color, stalk-shape, stalk-root,
stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,
stalk-color-below-ring, veil-type, veil-color, ring-number, ring-type,
spore-print-color, population, habitat
```

---

## REPOSITORY STRUCTURE TO CREATE

```
Group_ID/
├── README.md
├── requirements.txt
├── .gitignore
├── setup_git.sh
├── push_update.sh
├── data/
│   ├── raw/                                    ← user places mushrooms.csv here
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

---

## SHARED STRUCTURE FOR ALL 6 MEMBER NOTEBOOKS

Every notebook in `notebooks/` must follow this same skeleton, so all 6 are self-contained and
comparable. Only Section 4 (the technique itself) differs per member.

**Section 1 — Introduction**
- Explain in 2–3 sentences what this member's technique is and why/when it's used.

**Section 2 — Load Raw Data**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/mushrooms.csv')
print(df.shape)
df.head()
```

**Section 3 — Handle Missing Values & Drop Useless Column**
```python
df.replace('?', np.nan, inplace=True)
print(df.isnull().sum())
print((df.isnull().sum() / len(df) * 100).round(2))

# veil-type has only 1 unique value — no predictive information
df.drop(columns=['veil-type'], inplace=True)
```
Each member picks ONE simple, justified way to handle the `stalk-root` missing values
(mode imputation, 'unknown' category, or drop rows) and states the reason in a markdown cell.

**Section 4 — Apply This Member's Technique** (see per-member spec below)

**Section 5 — Encode Target Variable**
```python
df['class'] = df['class'].map({'e': 0, 'p': 1})
```

**Section 6 — EDA Visualization**
At least one chart illustrating the effect of this member's technique (e.g. before/after shape,
a heatmap of the encoded matrix, a bar chart of selected features). Save to
`../results/eda_visualizations/M<N>_<short_name>.png`

**Section 7 — Save Output**
Save the final processed dataframe to `../results/outputs/step_M<N>_<short_name>.csv`
(e.g. `step_M1_label_encoded.csv`). This filename must NOT be reused as an input by any other
member's notebook.

---

## PER-MEMBER TECHNIQUE SPECS (Section 4 of each notebook)

### Member 1 — IT0001_LabelEncoding.ipynb
Best for: tree-based models. Output columns: same (~21 cols after dropping veil-type).
- Apply `sklearn.preprocessing.LabelEncoder` to every remaining feature column.
- Print the letter→number mapping for each column.
- Show `df.head()` after encoding.
- Note in markdown: label encoding implies a false ordering, but tree-based models split on
  thresholds so this doesn't matter for them.

### Member 2 — IT0002_OneHotEncoding.ipynb
Best for: linear/NN models. Output columns: ~100+ cols.
- Apply `pd.get_dummies()` to all feature columns (`drop_first=True` optional, justify choice).
- Print shape before and after.
- Note in markdown: no false ordering is implied, but dimensionality grows a lot — good for
  linear models and neural networks, less ideal for distance-based models like KNN.

### Member 3 — IT0003_OrdinalImputation.ipynb
Best for: careful data cleaning focus. Output columns: same (~21 cols).
- Use `sklearn.preprocessing.OrdinalEncoder` instead of manual LabelEncoder.
- Spend extra depth on the imputation step from Section 3: compare mode-imputation vs a new
  'missing' category vs row-deletion side by side (counts, class-balance impact), and justify
  the final choice clearly — this is this member's main contribution.

### Member 4 — IT0004_BinaryEncoding.ipynb
Best for: balancing dimensionality. Output columns: ~50 cols.
- Use `category_encoders.BinaryEncoder` (install via `pip install category_encoders`) on all
  feature columns.
- Print shape before and after, and explain how binary encoding represents each category as a
  binary code across fewer columns than one-hot.

### Member 5 — IT0005_ChiSquareSelection.ipynb
Best for: reducing noise. Output columns: fewer.
- First label-encode features (needed for the chi-square test), then apply
  `sklearn.feature_selection.SelectKBest` with `chi2` against the target.
- Print each feature's chi-square score and p-value, ranked.
- Choose and justify a K (or a score threshold), then keep only the selected columns.
- Show shape before and after selection.

### Member 6 — IT0006_TargetFrequencyEncoding.ipynb
Best for: boosting models (XGBoost). Output columns: same (~21 cols).
- Implement both **frequency encoding** (replace each category with its occurrence count/rate)
  and **target encoding** (replace each category with the mean target value for that category —
  use a simple smoothed version or `category_encoders.TargetEncoder` to reduce leakage).
- Compare the two briefly and note the leakage risk of target encoding if not cross-validated.

---

## FILE — group_pipeline.ipynb (COMPARISON notebook, not a dependency chain)

This notebook does NOT feed any member's output into another. It only compares them.

**Section 1 — Introduction**
Explain that this notebook compares the 6 independent preprocessing outputs, not chains them.

**Section 2 — Load All 6 Outputs**
```python
import pandas as pd
files = {
    'Label Encoding': '../results/outputs/step_M1_label_encoded.csv',
    'One-Hot Encoding': '../results/outputs/step_M2_onehot_encoded.csv',
    'Ordinal + Imputation': '../results/outputs/step_M3_ordinal_encoded.csv',
    'Binary Encoding': '../results/outputs/step_M4_binary_encoded.csv',
    'Chi-Square Selected': '../results/outputs/step_M5_chi2_selected.csv',
    'Target/Frequency Encoding': '../results/outputs/step_M6_target_freq_encoded.csv',
}
dfs = {name: pd.read_csv(path) for name, path in files.items()}
for name, d in dfs.items():
    print(f"{name}: {d.shape}")
```

**Section 3 — Compare Shapes and Techniques**
Bar chart comparing column counts across all 6 outputs.
Save as `../results/eda_visualizations/GROUP_shape_comparison.png`

**Section 4 — Recommend Technique per Model**
A markdown table mapping each of the 6 planned models (Logistic Regression, Decision Tree,
Random Forest, SVM, KNN, Neural Network) to the preprocessing output best suited to it, with a
one-line reason (e.g. Neural Network → One-Hot Encoding; Random Forest → Label Encoding or
Chi-Square Selected).

**Section 5 — Train/Test Split for Each Recommended Pairing**
For each model, load its recommended preprocessing output and create an 80/20 train/test split
with `random_state=42`, `stratify=y`. Save each split to `../results/outputs/` with a clear
name, e.g. `X_train_labelenc.csv`, `X_test_labelenc.csv`, etc.

---

## FILE — README.md

Generate a professional README with:
- Project title and course info (IT2011, SLIIT, Year 2 Sem 1, 2026)
- Dataset description (UCI Mushroom, 8124 rows, 22 features, binary classification)
- Group member roles table with 6 rows: Member, IT Number, Preprocessing Technique, Model
- Explicitly state: "Each member's preprocessing notebook is independent and starts from the
  raw dataset — there is no dependency chain between member notebooks."
- Repository structure tree (from above)
- How to run: install requirements, then each member can run their own notebook in any order;
  `group_pipeline.ipynb` should be run last, after all 6 member outputs exist
- Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, category_encoders,
  tensorflow, nbformat

---

## FILE — requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
category_encoders
tensorflow
nbformat
jupyter
```

---

## FILE — .gitignore
```
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
Thumbs.db
.vscode/
.idea/
*.log
```
> NOTE: The dataset IS tracked by git (not ignored). All members get the data automatically
> when they clone the repo.

---

## FILE — setup_git.sh
```bash
#!/bin/bash
# Run this once to initialise git and push to GitHub
# Replace YOUR_GITHUB_USERNAME and YOUR_REPO_NAME before running

if [ -f "mushrooms.csv" ]; then
    mv mushrooms.csv data/raw/mushrooms.csv
    echo "Moved mushrooms.csv to data/raw/"
fi

git init
git add .
git commit -m "Initial commit — project structure, 6 independent preprocessing notebooks, dataset"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# Create all model branches from main
git checkout -b model/logistic-regression && git push origin model/logistic-regression && git checkout main
git checkout -b model/decision-tree       && git push origin model/decision-tree       && git checkout main
git checkout -b model/random-forest       && git push origin model/random-forest       && git checkout main
git checkout -b model/svm                 && git push origin model/svm                 && git checkout main
git checkout -b model/knn                 && git push origin model/knn                 && git checkout main
git checkout -b model/neural-network      && git push origin model/neural-network      && git checkout main
echo "All 6 model branches created and pushed. Share branch names with members."
```

## FILE — push_update.sh
```bash
#!/bin/bash
# Usage: bash push_update.sh "Added KNN model notebook"
MESSAGE=$1
git add .
git commit -m "${MESSAGE:-Update}"
git push origin main
```

---

## WORKFLOW SUMMARY

**PHASE 1 — Preprocessing (all 6 members work in PARALLEL, each independent):**
- Each member runs their own notebook directly against the raw CSV — no waiting on anyone else.
- All 6 push their notebook + output CSV to `main` (or their own short-lived branch merged
  back into `main`) whenever they're done, in any order.
- Once all 6 outputs exist in `results/outputs/`, anyone runs `group_pipeline.ipynb` to compare
  them and produce the train/test splits.

**PHASE 2 — Model Training (each member works on their own branch):**
- `model/logistic-regression` — Member 1
- `model/decision-tree` — Member 2
- `model/random-forest` — Member 3
- `model/svm` — Member 4
- `model/knn` — Member 5
- `model/neural-network` — Member 6

Each member clones/pulls `main`, checks out their branch, adds their model notebook to
`models/` using the train/test split `group_pipeline.ipynb` recommended for their model, pushes,
and opens a Pull Request into `main`.

---

## HOW TO RUN THIS PROMPT IN CLAUDE CODE

1. Create an empty folder on your PC named after your Group ID (e.g. `G01`)
2. Copy your `mushrooms.csv` into that folder (root — the script moves it)
3. Open Claude Code terminal inside that folder
4. Paste this entire prompt
5. Claude Code creates all folders, files, and the 6 independent notebooks automatically
6. Run `setup_git.sh` to push to GitHub and create the 6 model branches
7. Share the repo + branch names with your group — all 6 can start their preprocessing
   notebook immediately and in parallel, since none of them depend on each other
