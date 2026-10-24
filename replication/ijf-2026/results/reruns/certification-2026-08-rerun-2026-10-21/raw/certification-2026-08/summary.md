# Re-run of the August 2026 certification batch: certification-2026-08-rerun-2026-10-21

| comparison | instances |
|---|---|
| status same | 45 |
| status differs | 1 |
| ipopt agrees | 9 |
| ipopt differs | 0 |
| scip optimum agrees | 4 |
| scip optimum differs | 0 |
| ipopt - scip agrees | 32 |
| ipopt - scip differs | 0 |

## Table S5 (by the harness's cells; OPTIMAL / TIME_LIMIT / killed by the external clock)

| cell | record | re-run |
|---|---|---|
| A | 12/0/0 | 12/0/0 |
| B | 0/12/0 | 0/12/0 |
| C | 2/4/0 | 2/4/0 |
| D | 2/0/0 | 2/0/0 |
| E | 2/2/2 | 2/1/3 |
| F | 3/3/0 | 3/3/0 |
| G | 2/0/0 | 2/0/0 |
| all | 23/21/2 | 23/20/3 |

## Section 5.6, seasonal grid (cell B)

| quantity | record | re-run |
|---|---|---|
| s56_seasonal_ipopt_minus_scip_max | 2.520e-09 | 2.520e-09 |
| s56_seasonal_gap_min | 1.940e-04 | 1.732e-04 |
| s56_seasonal_gap_max | 4.900e-01 | 4.626e-01 |
| s56_seasonal_gap_median | 1.510e-02 | 5.860e-02 |

## Section 5.6 instances

| id | record Ipopt / SCIP | re-run Ipopt / SCIP / status / gap |
|---|---|---|
| 26 | 46.660913 / 45.314976 | 46.660913 / 45.314976 / TIME_LIMIT / 3.05e-01 |
| 44 | 57.868044 / 57.536983 | 57.868044 / 57.536983 / OPTIMAL / 1.09e-09 |
| 45 | 54.706978 / 53.681295 | 54.706978 / 53.681295 / OPTIMAL / 1.07e-08 |

Per instance: `comparison.csv`. Times and the gaps of time-limited instances depend on the machine and its load.
