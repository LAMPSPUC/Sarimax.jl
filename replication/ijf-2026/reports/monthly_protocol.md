# Table 3: the protocol of the archived M4 cells

Section 5.1 and Table S3 (`tab:m4_baseline`) describe the comparison with `forecast::auto.arima`
as run with a 120 s limit per fit and with stationarity checked after estimation (Table S3 rows
"Per-series time limit: 120" and "Admissibility: post-fit root checks, margin 1e-2"). The
archived cells that Table 3 prints were produced otherwise.

## What the archived runs used

Campaign A (`m4_benchmark`): monthly at commit `3d7f883` (48,000 series), the other five
frequencies at `aa68d57`. Its runner states every keyword explicitly
(`original-scripts/m4-recreio-2026-08/scripts/run_m4.jl`, function `m4Config`, bundle part 1):

| setting | archived value | Section 5.1 / Table S3 |
|---|---|---|
| per-fit time limit | `maxTimeSeconds = CAP`, with `CAP = nothing` unless a fifth argument is given; the campaign ran without it: "`maxTimeSeconds = nothing` (**no cap**)" (`REPRODUCE.md` of campaigns A-F, section A; the per-frequency table says "n/a — no cap" for every frequency) | 120 s |
| AR stationarity | `stationary = true`: imposed during estimation by the reflection parameterization, domain margin `1e-6` | checked after estimation |
| MA invertibility | `invertible = false`: coefficients in their box, not imposed | checked after estimation |
| post-fit checks | `assertStationarity = true`, `assertInvertibility = true`, `rootMargin = 1e-2`: candidates with a root within `1e-2` of the unit circle are rejected | post-fit root checks, margin `1e-2` |
| guard | `requireTermsWhenOverDifferenced = false` | not stated |

The 120 s limit belongs to another campaign: campaign B (objectives on M4 monthly, a 4,000-series
sample) ran with `maxTimeSeconds = 120.0` (same `REPRODUCE.md`, section B).

## How much it matters

* Time limit: in campaign A, 16 monthly fits ran longer than 120 s (`REPRODUCE.md`: "fits
  exceeding 120 s: hourly 22 (5.3%), quarterly 17, monthly 16, daily 3, yearly 1, weekly 0").
  With a 120 s limit those fits would have stopped early; nothing was censored in the archived
  run.
* Stationarity: the controlled objective campaign follows the Section 5.1 description apart from
  the time limit (AR in its box, roots checked after estimation, no cap; release `14dd581`). Its
  MSE OWA is 0.9039 monthly against Table 3's 0.9036; under the Table 3 protocol on the same code
  and environment, the weekly cell reproduces Table 3 on 359/359 series (0.8977), and the Section
  5.1 protocol gives 0.9002. The two treatments are different estimator configurations, and the
  differences are small but not zero.

## Minimal correction (for the authors)

Keep the Table 3 numbers (they are archived and re-executable) and describe their protocol:

* **Table S3:** "Per-series time limit: none" (instead of 120); "Admissibility: AR stationarity
  imposed by the reflection parameterization (margin 10^-6), MA coefficients in their box;
  post-fit root checks of both polynomials, margin 10^-2".
* **Section 5.1:** replace the sentence on stationarity with the wording proposed in
  `derived/m4-objectives-controlled-2026-10/manuscript_update_proposal.md` §5 (bundle part 2):

  > For the comparison with `forecast::auto.arima` (Tables 3 and 4), stationarity of the AR
  > polynomials is imposed during estimation through the reflection-coordinate parameterization
  > (domain margin 10^-6), and the MA coefficients are estimated in their finite box. For the
  > comparison of estimation objectives (Tables 5 and 6, Figure 3), stationarity and
  > invertibility are checked after estimation rather than imposed. In both cases every
  > candidate's complete AR and MA polynomials are checked after estimation, and candidates with
  > a root within 10^-2 of the unit circle are discarded, as in `forecast::auto.arima`.

  and remove the 120 s limit from the description of the Table 3 runs (no per-fit time limit was
  applied).

The alternative, making the text true by re-running Table 3 under the Section 5.1 protocol with a
120 s limit, would be a new campaign of the size of campaign A (about 82 CPU-hours for the six
frequencies of Sarimax.jl at the archived speed) and is not proposed.
