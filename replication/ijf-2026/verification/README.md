# Verification

| path | content |
|---|---|
| `VERIFICATION_REPORT.md` | what was verified for the submitted manuscript, how, and what remains |
| `ma_normalization/check_ma_normalization.py` | numerical check of the moving-average normalization of Supplementary Section S2 (the reflection form, the resultant identity, and the factor the package omits); numpy only |
| `ma_normalization/package_factor.jl` | the normalization factor the package applies, measured by fitting seasonal MA models with fixed coefficients; `result_14dd581.txt`, `result_21903b5.txt` |
| `defect_tests/` | targeted tests of the stepwise incumbent, the drift level in the ranking likelihood and the ranking likelihood with regressors at a given revision; `result_14dd581.txt` |

The Julia scripts take the path of a checkout of the revision to test:

```
julia +1.10.12 verification/ma_normalization/package_factor.jl PATH_TO_CHECKOUT
julia +1.10.12 verification/defect_tests/run.jl PATH_TO_CHECKOUT
python verification/ma_normalization/check_ma_normalization.py
```

They resolve their dependencies afresh in a temporary environment (they are not part of the
locked replication environment) and do not change the estimator.
