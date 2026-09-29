# Loan Default Scorecard

An end-to-end credit-risk pipeline that builds a probability-of-default (PD) scorecard on a
public loan dataset. It is also a case study in finding and removing data leakage.

A naive model on this data scores a test AUC of **1.00**, and all of that is leakage. v0.1 of this
project removed the obvious leaks and reported AUC 0.82 (with an unreproducible "≈0.74 honest"
figure). An independent audit ([`../AUDIT.md`](../AUDIT.md)) showed that both numbers were still
inflated. **v0.2** rebuilds the pipeline to fix every finding and reports a reproducible
**test AUC of 0.726 [95% CI 0.718–0.734]**, with one caveat that no cleaning can remove (below).

---

## Key findings

| Finding | Detail |
|---|---|
| Label proxies | A systematic scan (train only) flags every group with a ≥99% or ≤1% default rate: blank `rate_of_interest` / `Interest_rate_spread` / `Upfront_charges`, blank `property_value`/`LTV`, blank `age` / `submission_of_application`, and `credit_type == EQUI`. Rate columns are dropped; the other records are excluded from the scored population (routed to manual review) instead of being imputed. |
| Imputation leak (v0.1) | Mean-imputing `LTV`/`dtir1` put every leaked record on one value, which carried the leak into those features. v0.2 does no imputation: blanks get an explicit `MISSING` bin. |
| Variable selection | IV screen on train, then a redundancy screen (`business_or_commercial` ≡ `loan_type`, `loan_amount` ~ `property_value`, shared `income`/`dtir1` blanks), then forward selection on **validation** AUC that accepts a feature only if every coefficient keeps the correct sign. |
| Inert feature | `Credit_Score` has a flat default rate across deciles (IV ≈ 0.0007), which strongly suggests it is synthetic. |
| **Residual caveat** | In the scored population, `Status` matches "`rate_of_interest` is blank" **exactly** (correlation 1.000). The label may record a data-collection gap rather than a repayment failure. The pipeline is sound; this dataset cannot validate a credit model. |

## Performance (v0.2, complete applications only)

| Metric | Validation (n = 26,721) | Test (n = 26,651) | v0.1 test |
|---|---|---|---|
| AUC-ROC | 0.719 [0.711, 0.727] | **0.726 [0.718, 0.734]** | 0.822 (leak-inflated) |
| Gini | 0.438 [0.421, 0.455] | **0.452 [0.437, 0.469]** | 0.644 |
| KS | 0.323 [0.310, 0.339] | **0.326 [0.315, 0.345]** | 0.500 |
| Calibration | — | max decile gap 2.6 pp | "near-perfect" (incl. the PD≈1 leak block) |
| Records scored at PD > 0.99 | — | 0.0% | 10.1% |
| Coefficients with reversed sign | — | 0 | 3 |

Intervals are 95% bootstrap CIs (1,000 resamples). Selected features (in order of entry): `income`,
`LTV`, `Neg_ammortization`, `lump_sum_payment`, `dtir1`, `submission_of_application`,
`co-applicant_credit_type`, `loan_type`.

---

## Repository structure

```
loan-default-scorecard/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── Loan_Default_Score_Engine.ipynb   # full pipeline, executed with outputs
├── data/
│   └── README.md                         # how to obtain Loan_Default.csv (ignored by git)
└── docs/
    ├── Step_by_Step_Guide.docx           # what happened at each stage, and why
    └── Model_Validation_Report.docx      # independent-review write-up (v0.2)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place Loan_Default.csv in data/ (see data/README.md), then:
jupyter notebook notebooks/Loan_Default_Score_Engine.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/Loan_Default_Score_Engine.ipynb
```

The notebook finds the CSV automatically in `../data/`, `data/` or the working directory. A full
run takes about a minute.

## Pipeline at a glance

1. **Inspect**: shape, dtypes, target balance, missingness. Drop `ID` (row number) and `year` (constant).
2. **Split first**: stratified 60/20/20, before anything looks at the target.
3. **Label-proxy scan** (train): drop the rate columns; exclude blank-field and EQUI records; re-scan is clean.
4. **Clean**: fix LTV decimal errors. No imputation, no row drops, no winsorising (a no-op under quantile bins).
5. **EDA** (train): default rates by category; distributions by class.
6. **Bin + WoE + IV** (train): quintiles plus a `MISSING` bin; IV on the model's own bins; IV > 0.5 stops the run.
7. **Variable selection**: IV ≥ 0.02 → redundancy screen → sign-constrained forward selection on validation.
8. **Model**: WoE logistic regression; every coefficient negative (WoE = ln(good/bad)).
9. **Scale**: integer points (600 at 50:1, PDO 20).
10. **Evaluate** (test, once): AUC / Gini / KS with bootstrap CIs, decile calibration, leak checks.

## Notes & caveats

- **Target validity.** See the residual caveat above. Metrics describe how well application data predicts
  *this* label, which may be a data-collection artifact rather than borrower credit risk.
- **Scope.** 10.3% of applications (those with blank-field proxies) are outside the scored population.
- **Single vintage.** All data is 2019, so no out-of-time validation or drift assessment is possible.
- **Educational use.** Not for production credit decisioning.

## License

MIT — see [LICENSE](LICENSE).
