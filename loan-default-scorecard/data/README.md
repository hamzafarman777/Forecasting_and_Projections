# Data

The dataset (`Loan_Default.csv`) is **ignored by git** (`data/*.csv` in `.gitignore`). A copy ships
inside `loan-default-scorecard.zip` at the repository root. Unzip it, or drop your own download here.

## What you need

- **File:** `Loan_Default.csv`
- **Shape:** 148,670 rows × 34 columns
- **Vintage:** 2019 (single year)
- **Target:** `Status` (1 = default, 0 = non-default), base rate ≈ 24.6%

## How to use it

Place `Loan_Default.csv` in this `data/` folder. The notebook looks for it in `../data/`, `data/` and
the working directory, so no path edit is needed.

## Note

This is a public educational loan-default dataset, intended for learning and research, not for
production credit decisioning. It contains several forms of leakage through blank fields, which the
pipeline detects and excludes. Even after that, the label matches "`rate_of_interest` is blank" exactly
in the scored population. See the project README.
