# Preprocessing Outputs Audit Report

Audited all 6 notebooks/outputs against the expected shapes in the assignment reference table.
Raw dataset: `data/raw/mushrooms.csv`, 8,124 rows, 23 columns (22 features + `class`), with
`veil-type` correctly dropped by all 6 notebooks (1 unique value, no predictive information).

## IT0001 — Label Encoding
- Expected: 22 cols (21 features + 1 target)
- Actual (original): (8124, 22)
- Actual (fixed): unchanged
- **No issue found.** Shape matches, 0 missing values, all classes correctly mapped.

## IT0002 — One-Hot Encoding
- Expected: 96 cols exactly (21 features → 95 dummy columns with `drop_first=True`, computed
  from each feature's true cardinality including the explicit `'unknown'` category added for
  `stalk-root`'s missing values, + 1 `class` column)
- Actual (original/broken): (8124, 95) — **1 column short**
- Actual (fixed): (8124, 96)
- **Bug found and fixed.** Section 3 used the chained-assignment anti-pattern
  `df['stalk-root'].fillna('unknown', inplace=True)`. On this environment's pandas version
  (3.0.5, Copy-on-Write always on), that line is silently a no-op — it never wrote back to
  `df`, so `stalk-root` still had 2,480 `NaN` values when `pd.get_dummies()` ran. `pd.get_dummies`
  does not create a dummy column for `NaN`, so instead of getting its own explicit
  `stalk-root_unknown` column (as the notebook's own markdown documents), all 2,480 missing rows
  were silently folded into the same all-zero baseline as the 3,776 rows with `stalk-root == 'b'`
  (6,256 = 3,776 + 2,480, confirmed against the resulting all-zero row count). This mixed two
  semantically different things (a real category and "we don't know") into one indistinguishable
  encoding, and produced one fewer column than the documented design.
  **Fix:** changed the line to `df['stalk-root'] = df['stalk-root'].fillna('unknown')` (direct
  assignment instead of chained `inplace=True`). Notebook was re-run end-to-end; the corrected
  output now has its own `stalk-root_unknown` dummy column and the full 96-column shape, and
  `results/outputs/step_M2_onehot_encoded.csv` was overwritten with the corrected data.

## IT0003 — Ordinal Encoding + Imputation
- Expected: 22 cols
- Actual: (8124, 22)
- **No issue found.** Already uses the safe, non-chained `df['stalk-root'] = df['stalk-root'].fillna('missing')`
  pattern. 0 missing values, `OrdinalEncoder` correctly fit on all 21 feature columns.

## IT0004 — Binary Encoding
- Expected: ~50 cols (approximate in the assignment table)
- Actual: (8124, 64)
- **No issue found.** 64 is the mathematically correct output: `category_encoders.BinaryEncoder`
  needs `ceil(log2(n_categories + 1))` bits per feature; summing that across all 21 features
  (accounting for each feature's real cardinality) gives exactly 63 bits + 1 `class` column = 64.
  The ~50 figure in the reference table was a rough estimate, not a strict bound — verified the
  actual column count against the category-count math for every one of the 21 features and it
  is correct. No target leakage, `class` untouched and correctly appended at the end.

## IT0005 — Chi-Square Feature Selection
- Expected: fewer than 22 cols
- Actual: (8124, 11)
- **No issue found.** `SelectKBest(chi2)` scores are computed correctly, `K = 10` is applied
  (`df = df_le[top_features + ['class']]`), giving 10 selected features + `class` = 11 columns.
  Selection is genuinely applied to filter `df`, not just computed and discarded.

## IT0006 — Target / Frequency Encoding
- Expected: 22 cols
- Actual: (8124, 22)
- **No issue found.** Both frequency and target encodings are computed side-by-side for
  comparison; the notebook explicitly reassigns `df = df_target` before saving, so the saved
  output is the (documented, leakage-flagged) target-encoded version with no leftover duplicate
  columns from the comparison step. 0 missing values.

## Summary

| Notebook | Technique | Expected | Original Shape | Fixed Shape | Status |
|---|---|---|---|---|---|
| IT0001 | Label Encoding | 22 cols | (8124, 22) | — | OK |
| IT0002 | One-Hot Encoding | 96 cols | (8124, 95) | (8124, 96) | **Fixed** |
| IT0003 | Ordinal + Imputation | 22 cols | (8124, 22) | — | OK |
| IT0004 | Binary Encoding | ~50 cols (64 exact) | (8124, 64) | — | OK |
| IT0005 | Chi-Square Selection | <22 cols | (8124, 11) | — | OK |
| IT0006 | Target/Frequency Encoding | 22 cols | (8124, 22) | — | OK |

Only `notebooks/IT0002_OneHotEncoding.ipynb` and `results/outputs/step_M2_onehot_encoded.csv`
were modified. No other notebooks or outputs were changed.
