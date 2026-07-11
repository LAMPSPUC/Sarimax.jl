#!/usr/bin/env Rscript
# Block 1 - Validation (R forecast::Arima baseline). Writes JSONL run records.
suppressMessages({library(forecast); library(jsonlite)})

here <- tryCatch(dirname(sub("^--file=", "",
  commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])),
  error = function(e) ".")
raw  <- normalizePath(file.path(here, "..", "results", "raw"))
data <- file.path(raw, "validation", "data")
pkg_datasets <- normalizePath(file.path(here, "..", "..", "datasets"))
out  <- file.path(raw, "validation", "r_forecast_results.jsonl")
if (file.exists(out)) file.remove(out)

append_rec <- function(rec) cat(toJSON(rec, auto_unbox = TRUE, null = "null"),
                                "\n", file = out, append = TRUE, sep = "")

run_one <- function(dataset, y, order, seasonal = c(0, 0, 0), period = 1, xreg = NULL) {
  rec <- list(block = "validation", implementation = "R-forecast",
              dataset = dataset,
              order = sprintf("(%d,%d,%d)(%d,%d,%d)_%d", order[1], order[2], order[3],
                              seasonal[1], seasonal[2], seasonal[3], period),
              objective = "ml", solver = "css-ml", seed = 1234)
  res <- tryCatch({
    t0 <- Sys.time()
    if (period > 1) y <- ts(y, frequency = period)
    fit <- Arima(y, order = order,
                 seasonal = list(order = seasonal, period = period),
                 include.mean = TRUE, xreg = xreg, method = "ML")
    rec$runtime_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rec$status <- "ok"
    rec$estimates <- as.list(coef(fit))
    rec$loglike <- as.numeric(fit$loglik)
    rec$aic <- as.numeric(fit$aic)
    rec$bic <- as.numeric(fit$bic)
    rec$rss <- as.numeric(sum(residuals(fit)^2))
    rec
  }, error = function(e) {
    rec$status <- "failed"; rec$error <- conditionMessage(e); rec
  })
  append_rec(res)
  cat(dataset, res$order, "->", res$status, "\n")
}

arma <- read.csv(file.path(data, "sim_arma.csv"))$value
sx   <- read.csv(file.path(data, "sim_sarimax.csv"))
airp <- read.csv(file.path(pkg_datasets, "airpassengers.csv"))
airp_y <- airp[[2]]

run_one("sim_arma", arma, c(1, 0, 0))
run_one("sim_arma", arma, c(0, 0, 1))
run_one("sim_arma", arma, c(1, 0, 1))
run_one("airpassengers", airp_y, c(1, 0, 1))
run_one("airpassengers", airp_y, c(1, 0, 1), seasonal = c(1, 0, 1), period = 12)
# sim_sarimax is an ARX (dynamic-regression) DGP; fit the comparable ARX form
# (lagged y as a regressor, order (0,0,0)) to match SARIMAX.jl. See B4 / diagnose_exog.py.
run_arx_sim_sarimax <- function(sx) {
  y <- sx$value
  n <- length(y)
  ylag <- c(NA, y[1:(n - 1)])
  ok <- !is.na(ylag)
  xreg <- cbind(x1 = sx$x1[ok], x2 = sx$x2[ok], ar1 = ylag[ok])
  rec <- list(block = "validation", implementation = "R-forecast",
              dataset = "sim_sarimax", order = "ARX(1)+2exog",
              objective = "ls", solver = "Arima-xreg", seed = 1234,
              model_family = "ARX")
  res <- tryCatch({
    t0 <- Sys.time()
    fit <- Arima(y[ok], order = c(0, 0, 0), xreg = xreg, include.mean = TRUE)
    rec$runtime_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    rec$status <- "ok"; rec$estimates <- as.list(coef(fit))
    rec$loglike <- as.numeric(fit$loglik); rec$aic <- as.numeric(fit$aic)
    rec$bic <- as.numeric(fit$bic); rec$rss <- as.numeric(sum(residuals(fit)^2)); rec
  }, error = function(e) { rec$status <- "failed"; rec$error <- conditionMessage(e); rec })
  append_rec(res)
  cat("sim_sarimax ARX ->", res$status, "\n")
}
run_arx_sim_sarimax(sx)
cat("validation_r DONE ->", out, "\n")
