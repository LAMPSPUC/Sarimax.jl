# The protocols of the monthly M4 experiments

Two protocols ran on the M4 monthly series, and the manuscript describes them separately
(Section 5.1): the six-frequency comparison with `forecast::auto.arima` (Table 3), without a
per-series time cap and with AR stationarity imposed by reflection coordinates, and a separate
120-second-per-fit monthly experiment (Table S3). This note sets each against its archived
configuration.

## The six-frequency comparison (Table 3)

Campaign A (`m4_benchmark`): monthly at commit `3d7f883` (48,000 series), the other five
frequencies at `aa68d57`. Its runner states every keyword explicitly
(`original-scripts/m4-recreio-2026-08/scripts/run_m4.jl`, function `m4Config`, bundle part 1):

| setting | archived value | Section 5.1 |
|---|---|---|
| per-fit time limit | `maxTimeSeconds = CAP`, with `CAP = nothing` unless a fifth argument is given; the campaign ran without it: "`maxTimeSeconds = nothing` (**no cap**)" (`REPRODUCE.md` of campaigns A-F, section A; the per-frequency table says "n/a — no cap" for every frequency) | no per-series time cap |
| AR stationarity | `stationary = true`: imposed during estimation by the reflection parameterization, domain margin `1e-6` | reflection coordinates, margin 10^-6 |
| MA invertibility | `invertible = false`: coefficients in their box, not imposed | MA coefficients in finite boxes |
| post-fit checks | `assertStationarity = true`, `assertInvertibility = true`, `rootMargin = 1e-2`: candidates with a root within `1e-2` of the unit circle are rejected | AR/MA root checks at margin 10^-2 |
| over-differencing requirement | `requireTermsWhenOverDifferenced = false` | none |

Every row agrees (textual claims C2 and C4).

## The 120-second monthly experiment (Table S3)

The paired 1,500-series initialization comparison ran the runner archived as
`results/reruns/gate-1500-f90e554-2026-10-22/received/scripts/m4_gate1500.jl`; its
`:innovations` arm has the same per-series accuracy as the 48,000-series monthly census on those
series. The runner calls `auto` with `objectiveFunction = "mse"`, `searchMethod = "stepwise"`,
`integrationTest = "kpssShort"`, `seasonalIntegrationTest = "seas"`, the default ranking
criterion (AICc), `maxTimeSeconds = 120.0`, `stationary = invertible = false` with
`assertStationarity = assertInvertibility = true` and `rootMargin = 1e-2`,
`warmStartFromBox = true`, `requireTermsWhenOverDifferenced = true`, and the initialization of
the arm (`:innovations` or `:penalized`). Table S3 lists all of these except the warm start from
the box and the over-differencing requirement; the second changes which orders the search may
select (textual claim C3).

## How much the protocol matters

* **Time limit.** In campaign A, 16 monthly fits ran longer than 120 s (`REPRODUCE.md`: "fits
  exceeding 120 s: hourly 22 (5.3%), quarterly 17, monthly 16, daily 3, yearly 1, weekly 0").
  With a 120 s limit those fits would have stopped early; nothing was censored in the archived
  run.
* **Stationarity.** The controlled objective comparison checks roots after estimation instead
  of imposing stationarity (release `14dd581`, no time cap). Its MSE OWA is 0.9039 monthly
  against Table 3's 0.9036. On the weekly series, the same code and environment under the
  reflection-coordinate protocol reproduce Table 3 on 359 of 359 series (0.8977), against 0.9002
  with post-fit checks (Table S4). The two treatments are different estimator configurations,
  and the differences are small but not zero.
