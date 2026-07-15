#!/usr/bin/env Rscript
# R dependencies for the SARIMAX.jl experiment artifact.
# Reproducibility is obtained by installing from a DATED Posit Package Manager (PPM) snapshot,
# which serves the package versions that were current on that date. The run used R 4.6.1 with
# forecast 9.0.2 and jsonlite 2.0.0; the snapshot date below reproduces those versions.
#
# Usage:  Rscript experiments/scripts/setup/install_r.R
snapshot <- "2026-06-29"
options(repos = c(CRAN = paste0("https://packagemanager.posit.co/cran/", snapshot)))

pkgs <- c("forecast", "jsonlite")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

# Report the resolved versions for the reproducibility log.
cat("R", as.character(getRversion()), "\n")
for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")
# Expected: forecast 9.0.2, jsonlite 2.0.0 (snapshot ", snapshot, ").
