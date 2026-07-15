#!/usr/bin/env Rscript
# Block 2 - Rolling-origin forecasting (R forecast::Arima baseline).
# Expanding window on AirPassengers (s=12) and GDPC1 (quarterly). Refit at each origin.
suppressMessages({library(forecast); library(jsonlite)})

here <- tryCatch(dirname(sub("^--file=", "",
  commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])),
  error = function(e) ".")
raw <- normalizePath(file.path(here, "..", "results", "raw"))
pkg <- normalizePath(file.path(here, "..", "..", "datasets"))
out <- file.path(raw, "forecasting", "r_forecast_results.jsonl")
if (file.exists(out)) file.remove(out)

append_rec <- function(rec) cat(toJSON(rec, auto_unbox = TRUE, null = "null"),
                                "\n", file = out, append = TRUE, sep = "")

rolling_origins <- function(n, init_frac, H, step) seq(floor(init_frac * n), n - H, by = step)

metrics <- function(act, fc, train, m) {
  mae <- mean(abs(act - fc))
  denom <- if (length(train) > m) mean(abs(train[(m + 1):length(train)] -
                                          train[1:(length(train) - m)])) else NA
  c(mae = mae, rmse = sqrt(mean((act - fc)^2)),
    smape = mean(2 * abs(act - fc) / (abs(act) + abs(fc) + 1e-12)) * 100,
    mase = if (is.na(denom) || denom == 0) NA else mae / denom)
}

run_dataset <- function(dataset, y, m, H, step, init_frac, order, seasonal, period, label) {
  n <- length(y); origins <- rolling_origins(n, init_frac, H, step)
  acc <- list(); nfail <- 0
  for (k in origins) {
    train <- y[1:k]; act <- y[(k + 1):(k + H)]
    r <- tryCatch({
      t0 <- Sys.time()
      ts_train <- if (period > 1) ts(train, frequency = period) else ts(train)
      fit <- Arima(ts_train, order = order,
                   seasonal = list(order = seasonal, period = period), method = "ML")
      fc <- as.numeric(forecast(fit, h = H)$mean)
      el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      c(metrics(act, fc, train, m), elapsed = el)
    }, error = function(e) NULL)
    if (!is.null(r)) acc[[length(acc) + 1]] <- r else nfail <- nfail + 1
  }
  rec <- list(block = "forecasting", implementation = "R-forecast", dataset = dataset,
              order = label, horizon = H, seasonality = m, protocol = "rolling-origin",
              n_origins = length(origins), n_failures = nfail, seed = 1234)
  if (length(acc) > 0) {
    M <- do.call(rbind, acc)
    rec$status <- "ok"; rec$mae <- mean(M[, "mae"]); rec$rmse <- mean(M[, "rmse"])
    rec$smape <- mean(M[, "smape"]); rec$mase <- mean(M[, "mase"], na.rm = TRUE)
    rec$runtime_s <- sum(M[, "elapsed"])
  } else rec$status <- "failed"
  append_rec(rec)
  cat("R-forecast", dataset, "->", rec$status, "origins", length(origins), "fail", nfail, "\n")
}

airp <- read.csv(file.path(pkg, "airpassengers.csv"))[[2]]
gdp <- read.csv(file.path(pkg, "GDPC1.csv"), sep = ";")[[2]]

run_dataset("airpassengers", airp, 12, 12, 12, 0.7,
            c(1, 0, 1), c(1, 0, 1), 12, "(1,0,1)(1,0,1)_12")
run_dataset("GDPC1", gdp, 1, 8, 20, 0.7, c(1, 1, 1), c(0, 0, 0), 1, "(1,1,1) drift")
cat("forecasting_r (rolling-origin) DONE ->", out, "\n")
