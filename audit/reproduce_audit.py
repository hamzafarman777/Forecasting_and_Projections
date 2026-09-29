"""Reproduces every figure cited in AUDIT.md for the loan-default scorecard.

Run from the repo root:  python audit/reproduce_audit.py
Requires pandas, numpy, scikit-learn. Reads the CSV and notebook straight out of
loan-default-scorecard.zip, re-runs the notebook's modelling cells unchanged, then
runs the audit probes.
"""
import contextlib, io, json, warnings, zipfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
ZIP = 'loan-default-scorecard.zip'
zf = zipfile.ZipFile(ZIP)
RAW = pd.read_csv(zf.open('loan-default-scorecard/data/Loan_Default.csv'))


def section(t):
    print(f'\n=== {t} ===')


# ---------------------------------------------------------------- 1. label vs missingness
section('1. Default rate by missingness (raw data)')
for c in RAW.columns[RAW.isna().any()]:
    m = RAW[c].isna()
    print(f'{c:28s} missing={m.mean():6.3f}  P(def|missing)={RAW.Status[m].mean():.3f}  '
          f'P(def|present)={RAW.Status[~m].mean():.3f}')
rate_m, pv_m = RAW.rate_of_interest.isna(), RAW.property_value.isna()
equi = RAW.credit_type == 'EQUI'
flag = rate_m | pv_m | equi
print('LTV missing == property_value missing:', bool((RAW.LTV.isna() == pv_m).all()))
print(f'P(def | none of rate-missing/PV-missing/EQUI) = {RAW.Status[~flag].mean():.4f}  (n={int((~flag).sum())})')
print(f'Defaults with rate_of_interest present: {int((RAW.Status.eq(1) & ~rate_m).sum())} of {int(RAW.Status.sum())}')
print('EQUI rows with property_value missing:', int((equi & pv_m).sum()), 'of', int(equi.sum()))

# ---------------------------------------------------------------- 2. rerun the notebook
section('2. Notebook re-run (cells unchanged, plots skipped)')
nb = json.load(zf.open('loan-default-scorecard/notebooks/Loan_Default_Score_Engine.ipynb'))
code = '\n'.join(''.join(c['source']) for c in nb['cells']
                 if c['cell_type'] == 'code' and 'plt.subplots' not in ''.join(c['source']))
code = code.replace("pd.read_csv(DATA_PATH)", "RAW.copy()")
ns = {'RAW': RAW}
with contextlib.redirect_stdout(io.StringIO()):
    exec(code, ns)
g = ns
p, y_test, X_test, X_train, y_train = g['p'], g['y_test'], g['X_test'], g['X_train'], g['y_train']
Xw_tr, Xw_test, cols, woe = g['Xw_tr'], g['Xw_test'], g['cols'], g['woe']
print(f'test AUC={roc_auc_score(y_test, p):.4f}  (notebook/report: 0.822)')
print('coefficients (WoE convention: every sign should be negative):')
for k, v in g['coef'].items():
    print(f'  {k:28s} {v:+.3f}{"   <-- sign reversed" if v > 0 else ""}')
print(f"LTV train mean-imputation value lands in bin (58.97, 72.49]: WoE = {woe['LTV'][[k for k in woe['LTV'] if str(k).startswith('(58')][0]]:.2f}")
print(f"property_value MISSING WoE = {woe['property_value']['MISSING']:.2f}")
print(f'share of test scored PD > 0.99: {(p > 0.99).mean():.3f}')
print('WoE-feature correlation business_or_commercial ~ loan_type:',
      round(Xw_tr['business_or_commercial'].corr(Xw_tr['loan_type']), 3))

# ---------------------------------------------------------------- 3. leakage-adjusted variants
section('3. Leakage-adjusted AUC variants (test set)')


def fit_auc(c, Xtr, Xte, ytr, yte):
    m = LogisticRegression(max_iter=1000).fit(Xtr[c], ytr)
    return roc_auc_score(yte, m.predict_proba(Xte[c])[:, 1])


Xtr2, Xte2 = Xw_tr.copy(), Xw_test.copy()
Xtr2.loc[X_train.property_value.isna(), 'property_value'] = 0
Xte2.loc[X_test.property_value.isna(), 'property_value'] = 0
trm, tem = X_train.property_value.notna(), X_test.property_value.notna()
variants = {
    'a) PV MISSING bin neutralised (WoE=0), refit': fit_auc(cols, Xtr2, Xte2, y_train, y_test),
    'b) property_value dropped': fit_auc([c for c in cols if c != 'property_value'], Xw_tr, Xw_test, y_train, y_test),
    'c) property_value, LTV, dtir1 dropped': fit_auc([c for c in cols if c not in ('property_value', 'LTV', 'dtir1')],
                                                     Xw_tr, Xw_test, y_train, y_test),
    'd) PV-present rows only (train and test)': fit_auc(cols, Xw_tr[trm], Xw_test[tem], y_train[trm], y_test[tem]),
    'e) shipped model scored on PV-present test rows': roc_auc_score(y_test[tem], p[tem.values]),
}
for k, v in variants.items():
    print(f'{k:50s} AUC={v:.4f}')

ids = g['df'].loc[X_test.index, 'ID']
rate_missing_te = RAW.set_index('ID').loc[ids, 'rate_of_interest'].isna().values
print('PV-present test rows: corr(Status, rate_of_interest missing) =',
      round(np.corrcoef(y_test[tem], rate_missing_te[tem.values])[0, 1], 4))

# ---------------------------------------------------------------- 4. calibration
section('4. Calibration by decile (test)')
cal = pd.DataFrame({'p': p, 'y': y_test.values})
cal['d'] = pd.qcut(cal.p.rank(method='first'), 10, labels=False)
print(cal.groupby('d').agg(pred=('p', 'mean'), obs=('y', 'mean')).round(3).T.to_string())

# ---------------------------------------------------------------- 5. naive model
section('5. Naive model: every column, WoE with MISSING bins')
Xn = RAW.drop(columns=['Status', 'ID', 'year'])
a, b, ya, yb = train_test_split(Xn, RAW.Status, test_size=.2, stratify=RAW.Status, random_state=42)
A, B = pd.DataFrame(index=a.index), pd.DataFrame(index=b.index)
for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]) and a[c].nunique() > 20:
        q = np.unique(np.quantile(a[c].dropna(), np.linspace(0, 1, 6))); q[0], q[-1] = -np.inf, np.inf
        s1, s2 = g['num_bin'](a[c], q), g['num_bin'](b[c], q)
    else:
        s1, s2 = a[c].astype('object').fillna('MISSING'), b[c].astype('object').fillna('MISSING')
    w = g['woe_table'](s1, ya)
    A[c], B[c] = s1.map(w).astype(float), s2.map(w).astype(float)
m = LogisticRegression(max_iter=2000).fit(A.fillna(0), ya)
print(f'naive test AUC = {roc_auc_score(yb, m.predict_proba(B.fillna(0))[:, 1]):.4f}')

# ---------------------------------------------------------------- 6. row drops
section('6. Rows removed by dropna filters')
drop_cols = ['term', 'loan_limit', 'approv_in_adv', 'submission_of_application', 'age', 'loan_purpose', 'Neg_ammortization']
dm = RAW[drop_cols].isna().any(axis=1)
print(f'dropped {int(dm.sum())} rows, default rate {RAW.Status[dm].mean():.3f} vs {RAW.Status.mean():.3f} overall')
