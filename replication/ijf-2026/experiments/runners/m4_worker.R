# Worker process of the M4 campaign runner for the R reference methods (same JSON-lines
# protocol as m4_worker.jl):
#   stdin  <- {"series": 17, "y": [...], "s": 12, "H": 18}
#   stdout -> {"series": 17, "status": "OK", "forecast": [...], ...}
#
# Methods, exactly as the archived ForecastTester baselines ran them:
#   auto_arima  forecast::auto.arima(ts(y, frequency = s)) with its defaults, point forecast
#               (src/models/AutoArimaR.jl);
#   naive2      the M4 "Naive2" benchmark: classical multiplicative decomposition when the
#               90% ACF seasonality test passes, naive forecast, reseasonalised
#               (src/models/Naive.jl, the M4 organisers' SeasonalityTest).
#
# Usage: Rscript --vanilla m4_worker.R <method>

suppressMessages({
  library(forecast)
  library(jsonlite)
})

method <- commandArgs(trailingOnly = TRUE)[[1]]
stopifnot(method %in% c("auto_arima", "naive2"))

SeasonalityTest <- function(input, ppy) {
  # Used to determine whether a time series is seasonal (M4 organisers' code).
  tcrit <- 1.645
  if (length(input) < 3 * ppy) {
    test_seasonal <- FALSE
  } else {
    xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
    clim <- tcrit / sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- (abs(xacf[ppy]) > clim[ppy])
    if (is.na(test_seasonal) == TRUE) test_seasonal <- FALSE
  }
  test_seasonal
}

naive2 <- function(y, s, H) {
  input <- ts(y, frequency = s)
  ST <- FALSE
  if (s > 1) ST <- SeasonalityTest(input, s)
  if (ST == TRUE) {
    dec <- decompose(input, type = "multiplicative")
    des_input <- input / dec$seasonal
    SIout <- head(rep(dec$seasonal[(length(dec$seasonal) - s + 1):length(dec$seasonal)], H), H)
  } else {
    des_input <- input
    SIout <- rep(1, H)
  }
  list(forecast = as.numeric(naive(des_input, h = H)$mean * SIout), seasonal = ST)
}

fit_auto <- function(y, s, H) {
  model <- auto.arima(ts(y, frequency = s))
  ord <- arimaorder(model)
  get <- function(k) if (k %in% names(ord)) unname(ord[[k]]) else 0L
  list(forecast = as.numeric(forecast(model, h = H)$mean),
       p = get("p"), d = get("d"), q = get("q"), P = get("P"), D = get("D"), Q = get("Q"),
       has_mean = as.integer("intercept" %in% names(coef(model))),
       has_drift = as.integer("drift" %in% names(coef(model))))
}

cat(toJSON(list(ready = TRUE, R = paste(R.version$major, R.version$minor, sep = "."),
                forecast = as.character(packageVersion("forecast"))), auto_unbox = TRUE), "\n", sep = "")
flush(stdout())

con <- file("stdin", open = "r")
while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
  if (!nzchar(trimws(line))) next
  job <- fromJSON(line)
  t0 <- Sys.time()
  out <- tryCatch({
    res <- if (method == "auto_arima") fit_auto(job$y, job$s, job$H) else naive2(job$y, job$s, job$H)
    c(list(status = "OK"), res)
  }, error = function(e) list(status = paste0("ERROR:", substr(gsub("\n", " ", conditionMessage(e)), 1, 300))))
  out$series <- job$series
  out$fit_seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 3)
  cat(toJSON(out, auto_unbox = TRUE, digits = NA), "\n", sep = "")
  flush(stdout())
}
