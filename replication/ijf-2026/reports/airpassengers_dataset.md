# The package's "AirPassengers" series

## Identity

`load_dataset(AIR_PASSENGERS)` reads `datasets/airpassengers.csv`: 204 monthly observations,
1991-07-01 to 2008-06-01, values 2.81452 to 29.665356 (sum 2181.66363476). It is the series
`a10` of the R package `fpp2` (monthly anti-diabetic drug sales in Australia: total scripts for
products under ATC code A10, from the Australian Health Insurance Commission). The comparison
was made against the CSV export of `fpp2::a10` published by the Rdatasets project
(`https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/a10.csv`, read on 2026-10-20): all 204
values agree to the printed digits, in the same order (July 1991 first). The file was added to
the repository on 2023-09-07 (commit `8abd5e1`).

It is not the Box-Jenkins airline series (`datasets::AirPassengers` in R: 144 monthly totals of
international airline passengers, January 1949 to December 1960, starting 112, 118, 132), which
the package's documentation named.

## Package change (commit "Say what the AIR_PASSENGERS file is, ...")

* The data are not replaced: `AIR_PASSENGERS` loads the same file as before, so no existing
  result changes.
* `ANTIDIABETIC_DRUG_SALES` is exported as the same `Datasets` value under a descriptive name.
* The docstring, the README and the tutorial say what the series is.
* `test/datasets.jl` pins the identity: 204 values, monthly dates from 1991-07-01 to 2008-06-01
  without gaps, the first four values, the minimum (2.81452, 1992-02-01), the maximum
  (29.665356, 2008-01-01) and the sum.

## Where the manuscript and the archive use it

Every experiment that loads `AIR_PASSENGERS` used `a10`, labelled "AirPassengers" or "airline":

* the July 2026 battery: validation (`run_validation_julia.jl`, the `airpassengers` rows),
  rolling-origin forecasting (`run_forecasting_*`), architecture checks
  (`run_architecture_extensions.jl`);
* Table S11 (`tab:multistart`), case "Airline model, log a10" (`efg_julia_experiments.jl`
  part G loads `log.(load_dataset(AIR_PASSENGERS))`);
* the README's "Verified against R" table (R was given the same file, so the comparison itself
  is valid; only the name is wrong).

The R and Python comparisons of the battery read the same CSV, so every package-to-package
agreement stands.

## In the manuscript

Table S11 names the case "Airline model, log a10", and the text below it states that the airline
specification is applied to the log of the `a10` series (Australian anti-diabetic drug sales),
not to the Box-Jenkins passenger data. This agrees with the file the experiments read.
