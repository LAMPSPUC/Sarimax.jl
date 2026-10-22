# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 16/08 03:40]
for (rep in 1:6) for (p in 1:2) {
  f <- sprintf("/tmp/m4s/nsser_%d_%d.csv", rep, p)
  if (!file.exists(f)) next
  y <- read.csv(f)$y
  fit <- tryCatch(arima(y, order=c(p,0,0), include.mean=FALSE, method="ML"), error=function(e) NULL)
  if (is.null(fit)) next
  cat(sprintf("rep=%d p=%d R_ML=%s\n", rep, p, paste(sprintf("%.6f", coef(fit)), collapse=",")))
}
