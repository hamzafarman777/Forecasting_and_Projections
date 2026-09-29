# Model Audit — Loan Default Scorecard v0.1

> **Update:** every finding below has been addressed in **v0.2**. See [Remediation status](#remediation-status-v02) at the end. The audited v0.1 notebook is kept unchanged at `audit/Loan_Default_Score_Engine_v0.1.ipynb`.

**Scope (v0.1):** `loan-default-scorecard.zip`: `notebooks/Loan_Default_Score_Engine.ipynb`, `docs/Model_Validation_Report.docx`, `docs/Step_by_Step_Guide.docx`, `README.md`, `data/Loan_Default.csv`.
**Method:** re-ran the notebook's cells unchanged (plots skipped) on the bundled CSV, then ran probes on the data and the model. Every number below comes from `python audit/reproduce_audit.py` (about 12 seconds).
**Reproduction:** the headline numbers match the report exactly: test AUC 0.8219, Gini 0.6439, KS 0.5002; validation AUC 0.8189.

## Verdict

The existing validation report rates the model **Conditional** because of one leak it has not fixed (the `property_value` MISSING bin). This audit finds a more basic problem. **`Status` in this dataset is not a credit outcome. It is almost fully determined by which fields were left blank.** So no model trained on it, the "leakage-adjusted" one included, can be shown to measure credit risk. The model's status should be **Not validated — data unfit for purpose**, not "Conditional". The notebook runs correctly and its calibration is genuine. Several statements in the report are wrong or cannot be reproduced (listed below).

---

## Critical findings

### C1. The target is a missing-data artifact, not default behaviour
| Condition (raw data, n = 148,670) | Default rate |
|---|---|
| `rate_of_interest` missing | **100.0%** |
| `rate_of_interest` present | **0.2%** (200 of 36,639 defaults) |
| `property_value` missing | **100.0%** |
| None of {rate missing, property_value missing, `credit_type`=EQUI} | **0.0000%** (n = 112,028) |

The notebook reads the rate-column leak only in one direction ("99.5% of defaults are missing the rate"). It misses the stronger fact: a loan with a recorded rate almost never defaults. Records with none of the three blank-field flags have zero defaults. Every positive label sits in a record with a data-collection gap. Dropping the leaking columns does not fix this, because the label itself was produced by the same process that left the fields blank.

**Consequence:** among test rows with `property_value` present, which is the population the "honest" model is judged on, `corr(Status, rate_of_interest is missing) = 1.0`. The "genuine signal" the model finds is how well the remaining features predict *which records have a blank interest-rate field*. The dataset cannot validate a credit-risk model. This is the finding the report most needs to state.

### C2. The leak also enters through the mean-imputed LTV and dtir1, not only the MISSING bin
`LTV` is missing on exactly the same rows as `property_value`. `dtir1` is missing on 15,096 of those 15,098 rows. The notebook mean-imputes both **before** binning. Every leaked row therefore gets the same value, and that value falls into one quantile bin:

- LTV mean 72.49 → bin (58.97, 72.49], WoE **−0.84**. Every other LTV bin is positive, so the curve is non-monotonic.
- dtir1 mean 37.73 → bin (31.0, 37.73], WoE **−0.71**. Same non-monotonic shape.

The report calls the high IVs for LTV and dtir1 (0.76 and 0.79) an "imputation artifact" that "inflates their apparent importance". In fact those bins are the `property_value` leak, copied. They add few points in the shipped model only because the MISSING bin already carries the signal. **If the MISSING bin is neutralised on its own, the leak comes back through these two features** (variant *a* below). The planned remediation (fix the `property_value` MISSING bin) is incomplete.

### C3. The "leakage-adjusted AUC ≈ 0.74" cannot be reproduced
Neither the notebook nor the docs show how 0.74 was calculated. Plausible remediations give:

| Variant (test set) | AUC |
|---|---|
| Shipped model | 0.822 |
| a) `property_value` MISSING WoE set to 0, refit | 0.792 (leak survives through LTV/dtir1, C2) |
| b) `property_value` dropped | 0.769 |
| c) `property_value`, LTV, dtir1 dropped | 0.726 |
| d) trained and tested only on rows with `property_value` present | 0.697 |
| e) shipped model scored on rows with `property_value` present | 0.697 |

The only leak-free comparison (d/e) gives about **0.70**, not 0.74. Given C1, even 0.70 measures how well the features predict the blank-rate artifact, not credit risk. The README, validation report, guide and notebook all repeat 0.74. It should be removed or replaced with a documented calculation.

---

## High findings

### H1. The claim that preprocessing was fitted on train only is false
The report (§3.2) and the guide (Step 7) say all imputation, scaling and binning were fitted on the training partition. In the notebook, these steps all run on the **full dataset before** `train_test_split`:
- mean imputation (cell 11)
- IQR winsorising caps (cell 13)
- the IV screen that picks the features (cell 20)

Only WoE binning is fitted on train. The numeric effect is probably small. The report still states it as a control that does not exist.

### H2. Coefficient signs reversed. The scorecard breaks the monotonicity rule
With WoE = ln(%good/%bad), every coefficient should be negative. Three are positive:
- `loan_amount` **+1.756**: the largest coefficient in the model, and reversed
- `co-applicant_credit_type` +0.223
- `LTV` +0.084

This comes from collinearity among the WoE inputs: income ~ loan_amount r = 0.54; LTV ~ property_value 0.50; Gender ~ co-applicant −0.54. A standard scorecard review would reject it: the points given to `loan_amount` bins run against what those bins show on their own.

### H3. `business_or_commercial` and `loan_type` are the same feature
Their WoE correlation is 0.979. In the raw data, `business_or_commercial == 'b/c'` holds on exactly the same 20,762 rows as `loan_type == 'type2'`. Both go into the model, which splits one effect across two coefficients and makes each unstable. Drop one.

### H4. "Four distinct leakage sources were identified and removed" (report §7) is overstated
Two were removed: the rate columns and `credit_type`. A third (`property_value` MISSING) was documented but left in. There is no fourth.

---

## Medium findings

- **M1. Biased row drop.** The `dropna` filters remove 4,728 rows with a 29.3% default rate, against 24.6% overall. Rows missing `age` or `submission_of_application` default at 100%, so this step quietly removes another leak without saying so.
- **M2. The calibration claim needs context.** Decile calibration really is tight (worst gap about 0.01). But the top decile is predicted at 0.999 and observed at 1.000. That decile is the leaked block (10.1% of test is scored PD > 0.99), so the headline "near-perfect" calibration partly measures the leak. Calibration on the rows with `property_value` present is also good (max gap about 0.01). That subset is the fair figure to report.
- **M3. The validation set is never used.** No hyperparameters, bins or thresholds are tuned on it. The 60/20/20 split therefore costs 20% of the training data and gives nothing back beyond a stability check.
- **M4. No uncertainty on the metrics.** AUC, Gini and KS are point estimates with no bootstrap confidence intervals.

## Low / documentation findings

- **L1.** The README points to `notebooks/Loan_Default_Pipeline.ipynb`. The shipped file is `Loan_Default_Score_Engine.ipynb`. The report and guide use the old name too.
- **L2.** The README and `data/README.md` say the dataset is "not committed". The zip includes the 28 MB `Loan_Default.csv`.
- **L3.** The notebook has no saved outputs. `DATA_PATH` defaults to the working directory, but the data lives in `../data/`.
- **L4.** The README understates the naive-model result as "AUC above 0.95". A WoE logistic regression on every raw column scores **AUC = 1.0000** on test.
- **L5.** `requirements.txt` has no upper bounds. The notebook runs on pandas 3.0 / scikit-learn 1.9, with deprecation warnings only.

## What is sound

- Cleaning logic (LTV decimal fix, winsorising instead of deleting) and the reasons given for it.
- Detection and removal of the rate columns and `credit_type`.
- `Credit_Score` really is inert: default rate is flat at 24–25% across every decile.
- The WoE and PDO arithmetic is correct: factor 28.85, offset 487.12, and the −194-point MISSING bin matches the report.
- Test-set calibration.

## Recommendations

1. **Change the status to "Not validated — data unfit for purpose".** State C1 in the executive summary. Do not describe any AUC on this data as "true discriminating power".
2. If the work goes on as a teaching exercise, treat blank-field patterns as label proxies across the whole dataset. Restrict to records with complete fields, or model the blank-rate flag openly as the target, and say that this is what is being modelled.
3. Fit imputation, winsorising and IV selection inside the training fold, or correct the text in §3.2.
4. Leave LTV/dtir1 missing (their own MISSING bin) instead of mean-imputing them, so the leak stays visible.
5. Remove one of `loan_type` / `business_or_commercial`. Enforce negative coefficients: drop or merge the offending features, or fit with a sign constraint.
6. Replace "≈ 0.74" with a documented, reproducible figure, with bootstrap confidence intervals.
7. Fix the notebook file name, the data-commit statement and the `DATA_PATH` default.

*The other notebooks in this repo (Barrick Gold forecast, Loan Default Pilot and Coursera) were not in scope.*

---

## Remediation status (v0.2)

The pipeline was rebuilt in `loan-default-scorecard/notebooks/Loan_Default_Score_Engine.ipynb`, which is committed with its outputs. The docs and README were rewritten to match.

| Finding | Status | What v0.2 does |
|---|---|---|
| C1 target is a missing-data artifact | **Mitigated, not fixable** | A label-proxy scan (train only) flags every group of ≥ 30 records with a default rate ≥ 99% or ≤ 1%. Rate columns are dropped. Records with blank `property_value`/`age`/`submission_of_application` or `credit_type = EQUI` are excluded from the scored population. The notebook states and prints the residual fact: in scope, `corr(Status, rate blank) = 1.000`. Status is set to "Not validated for credit use". |
| C2 leak via mean-imputed LTV/dtir1 | Fixed | No imputation anywhere; blanks get an explicit `MISSING` bin. Largest scored \|WoE\| is 2.22 (v0.1: 10.9). No test record scores PD > 0.99 (v0.1: 10.1%). |
| C3 unreproducible 0.74 | Fixed | Withdrawn. Every figure is computed in the notebook. Test AUC **0.726 [0.718, 0.734]**; 0.74 lies outside the CI. |
| H1 preprocessing on full data | Fixed | The split is the first step. Leak scan, bins, WoE, IV and selection are fitted on train only. |
| H2 reversed coefficient signs | Fixed | Forward selection accepts a feature only if every coefficient stays negative. 0 reversed signs. Cost: validation AUC 0.719 vs 0.722 unconstrained. |
| H3 duplicate feature | Fixed | A WoE redundancy screen (\|r\| > 0.7) drops `business_or_commercial` (r = 0.999 with `loan_type`) and `loan_amount` (r = 0.798 with `property_value`). A new shared-blank check neutralises `dtir1`'s `MISSING` bin (98.8% of its blanks are also `income` blanks), so that blank is not counted twice. |
| H4 "four leaks" overstated | Fixed | The report now lists exactly what the scan flags. |
| M1 biased row drop | Fixed | No rows dropped for blank categoricals; they become `MISSING`. The 100%-default `age`/`submission` blanks are excluded openly as label proxies. |
| M2 calibration included the leak | Fixed | Calibration is on the in-scope test set: max decile gap 2.6 pp. |
| M3 validation set unused | Fixed | Validation AUC drives variable selection; test is scored once. |
| M4 no uncertainty | Fixed | 1,000-resample bootstrap 95% CIs on AUC, Gini and KS. |
| L1–L3 docs/paths | Fixed | Correct notebook name everywhere; data described as git-ignored and bundled in the zip; `DATA_PATH` auto-detected. |
| L4 naive AUC understated | Fixed | README states naive AUC 1.00. |
| L5 unpinned requirements | Fixed | Upper bounds added; verified on pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1. |

**v0.2 results** (8 features: `income`, `LTV`, `Neg_ammortization`, `lump_sum_payment`, `dtir1`, `submission_of_application`, `co-applicant_credit_type`, `loan_type`):

| | Validation (n = 26,721) | Test (n = 26,651) |
|---|---|---|
| AUC | 0.719 [0.711, 0.727] | 0.726 [0.718, 0.734] |
| Gini | 0.438 [0.421, 0.455] | 0.452 [0.437, 0.469] |
| KS | 0.323 [0.310, 0.339] | 0.326 [0.315, 0.345] |

**Still open:**
- C1 cannot be fixed with this dataset; a genuine repayment outcome is needed.
- WoE is non-monotonic in LTV and dtir1 (accepted, but flagged for coarse-classing review).
- There is no out-of-time sample.
- The manual-review policy for the 10.3% of applications out of scope is undefined.
