#!/usr/bin/env Rscript
# Energy (PJME daily) rolling-origin forecasting - R forecast::Arima baseline.
# Same window/origins/order as the SARIMAX.jl energy script.
#
# Port of original-scripts/battery-2026-07/scripts/run_energy_forecasting_r.R. Window, origins,
# order, Arima(method = "ML"), metrics and record layout (jsonlite::toJSON with its default
# 4-digit rounding) unchanged. Adaptations:
#   * input: pjme_daily.csv derived in this run by setup/gen_energy_data.py from
#     $IJF_PJME_DIR/PJME_hourly.csv ($IJF_RUN_DIR/raw/battery-2026-07/energy/data/);
#   * output: $IJF_RUN_DIR/raw/battery-2026-07/energy/r_forecast_results.jsonl;
#   * the pinned R stack is R 4.4.1 + forecast 8.23.0 + jsonlite 1.8.9 (environments/r/renv.lock);
#     the archived run used R 4.6.1 + forecast 9.0.2 + jsonlite 2.0.0. No API used here differs.
suppressMessages({library(forecast); library(jsonlite)})

run_dir <- Sys.getenv("IJF_RUN_DIR")
if (!nzchar(run_dir)) stop("IJF_RUN_DIR is not set: the battery runners write only below $IJF_RUN_DIR/raw/battery-2026-07/")
raw <- file.path(normalizePath(run_dir, mustWork = TRUE), "raw", "battery-2026-07")
src <- file.path(raw, "energy", "data", "pjme_daily.csv")
if (!file.exists(src)) stop(src, " not found: it is derived in this run by experiments/runners/battery/setup/gen_energy_data.py, which needs ./reproduce.sh fetch pjme")
out <- file.path(raw, "energy", "r_forecast_results.jsonl")
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
if (file.exists(out)) invisible(file.remove(out))

WINDOW <- 540; M <- 7; H <- 14; STEP <- 28; INIT <- 0.7
rolling_origins <- function(n, init_frac, H, step) seq(floor(init_frac * n), n - H, by = step)
metrics <- function(act, fc, train, m) {
  mae <- mean(abs(act - fc))
  denom <- if (length(train) > m) mean(abs(train[(m + 1):length(train)] - train[1:(length(train) - m)])) else NA
  c(mae = mae, rmse = sqrt(mean((act - fc)^2)),
    smape = mean(2 * abs(act - fc) / (abs(act) + abs(fc) + 1e-12)) * 100,
    mase = if (is.na(denom) || denom == 0) NA else mae / denom)
}

yall <- read.csv(src)$value
y <- tail(yall, WINDOW); n <- length(y)
origins <- rolling_origins(n, INIT, H, STEP)
acc <- list(); rts <- c(); nfail <- 0
for (k in origins) {
  train <- y[1:k]; act <- y[(k + 1):(k + H)]
  r <- tryCatch({
    t0 <- Sys.time()
    fit <- Arima(ts(train, frequency = M), order = c(1, 1, 1),
                 seasonal = list(order = c(1, 0, 1), period = M), method = "ML")
    fc <- as.numeric(forecast(fit, h = H)$mean)
    rts <<- c(rts, as.numeric(difftime(Sys.time(), t0, units = "secs")))
    metrics(act, fc, train, M)
  }, error = function(e) { nfail <<- nfail + 1; NULL })
  if (!is.null(r)) acc[[length(acc) + 1]] <- r
}
rec <- list(block = "energy_forecasting", implementation = "R-forecast", dataset = "PJME_daily",
            order = "(1,1,1)(1,0,1)_7", horizon = H, seasonality = M, protocol = "rolling-origin",
            n_origins = length(origins), window = WINDOW, n_failures = nfail, seed = 1234)
if (length(acc) > 0) {
  Mx <- do.call(rbind, acc)
  rec$status <- "ok"; rec$mae <- mean(Mx[, "mae"]); rec$rmse <- mean(Mx[, "rmse"])
  rec$smape <- mean(Mx[, "smape"]); rec$mase <- mean(Mx[, "mase"], na.rm = TRUE); rec$runtime_s <- sum(rts)
} else rec$status <- "failed"
cat(toJSON(rec, auto_unbox = TRUE, null = "null"), "\n", file = out, append = TRUE, sep = "")
cat("R-forecast PJME_daily ->", rec$status, "origins", length(origins), "fail", nfail, "\n")
