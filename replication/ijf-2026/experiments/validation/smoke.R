# Smoke test, R side: the forecast::Arima baseline on the simulated ARMA fixture.
# Prints one JSON object per measured quantity (engine, test, quantity, value); the
# orchestrator compares them with smoke_expected.json, which holds the values the July 2026
# battery archived (R 4.6.1, forecast 9.0.2) for the same fits.
#
# Usage: Rscript --vanilla smoke.R <fixtures directory>

suppressMessages({
  library(forecast)
  library(jsonlite)
})

fixtures <- commandArgs(trailingOnly = TRUE)[[1]]
emit <- function(test, quantity, value, note = "") {
  cat(toJSON(list(engine = "r", test = test, quantity = quantity, value = value, note = note),
             auto_unbox = TRUE, digits = NA), "\n", sep = "")
}
run <- function(name, expr) {
  t0 <- Sys.time()
  tryCatch(expr, error = function(e) emit(name, "error", conditionMessage(e)))
  emit(name, "seconds", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2))
}

x <- read.csv(file.path(fixtures, "sim_arma.csv"))$value
emit("r_runtime", "R_version", paste(R.version$major, R.version$minor, sep = "."))
emit("r_runtime", "forecast_version", as.character(packageVersion("forecast")))

# 1. Conditional sum of squares: the convention SARIMAX.jl's :warmup reproduces.
run("r_ar1_css", {
  fit <- Arima(x, order = c(1, 0, 0), include.mean = TRUE, method = "CSS")
  emit("r_ar1_css", "ar1", unname(coef(fit)["ar1"]))
  emit("r_ar1_css", "intercept", unname(coef(fit)["intercept"]))
  emit("r_ar1_css", "rss", sum(residuals(fit)^2))
})

# 2. Exact Gaussian maximum likelihood, and its forecasts.
run("r_ar1_ml", {
  fit <- Arima(x, order = c(1, 0, 0), include.mean = TRUE, method = "ML")
  phi <- unname(coef(fit)["ar1"])
  mu <- unname(coef(fit)["intercept"])
  emit("r_ar1_ml", "ar1", phi)
  emit("r_ar1_ml", "intercept", mu)
  emit("r_ar1_ml", "loglik", as.numeric(fit$loglik))
  fc <- as.numeric(forecast(fit, h = 6)$mean)
  analytic <- mu + phi^(1:6) * (x[length(x)] - mu)
  emit("r_ar1_ml", "forecast_h6", fc[6])
  emit("r_ar1_ml", "forecast_matches_ar1_recursion", max(abs(fc - analytic)) < 1e-10)
})

# 3. Automatic order selection (the M4 baseline's entry point) on the same series.
run("r_auto_arima", {
  fit <- auto.arima(ts(x, frequency = 1))
  ord <- arimaorder(fit)
  emit("r_auto_arima", "order", paste(ord[c("p", "d", "q")], collapse = ","))
  emit("r_auto_arima", "aicc", as.numeric(fit$aicc))
})
