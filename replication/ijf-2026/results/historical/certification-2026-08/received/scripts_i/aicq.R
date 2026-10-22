# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 16/08 03:59]
# A fatia anterior congelava Q=0. Aqui a grade cobre a interacao PxQ, que e o suspeito nomeado.
suppressPackageStartupMessages(library(forecast))
train <- read.csv("/Users/davi/dev/ForecastTester/datasets/Monthly-train.csv", header=TRUE)
ids <- as.integer(readLines("/tmp/m4s/rand400.txt"))[1:100]
cat("sid,p,q,P,Q,npar,aicc\n", file="/tmp/m4s/rq_aicc.csv")
specs <- list()
for (pq in list(c(1,1), c(0,1))) for (P in 0:2) for (Q in 0:1)
  specs[[length(specs)+1]] <- c(pq[1], pq[2], P, Q)
for (i in ids) {
  y <- suppressWarnings(as.numeric(train[i,-1])); y <- y[!is.na(y)]
  if (length(y) < 60) next
  x <- ts(y, frequency=12)
  for (sp in specs) {
    r <- tryCatch({
      f <- Arima(x, order=c(sp[1],1,sp[2]),
                 seasonal=list(order=c(sp[3],0,sp[4]), period=12),
                 include.mean=FALSE, method="CSS-ML")
      sprintf("%d,%d,%d,%d,%d,%d,%.6f", i, sp[1],sp[2],sp[3],sp[4], length(coef(f))+1, f$aicc)
    }, error=function(e) NULL)
    if (!is.null(r)) cat(r, "\n", sep="", file="/tmp/m4s/rq_aicc.csv", append=TRUE)
  }
}
cat("ok\n")
