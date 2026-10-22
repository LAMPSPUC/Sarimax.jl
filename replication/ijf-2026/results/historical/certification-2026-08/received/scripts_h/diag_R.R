# [RECUPERADO DO TRANSCRIPT DA SESSAO EM 21/10/2026 — escrito 20/08 13:50]
suppressPackageStartupMessages(library(forecast))
T <- 180; H <- 12; S <- 12; K <- 3
for (tipo in c("erros_arima","ma_puro","armax")) {
  mp <- tipo == "ma_puro"
  ord <- if (mp) c(0,0,1) else c(1,0,1); sord <- if (mp) c(0,0,1) else c(1,0,0)
  R <- matrix(NA_real_, 20, H+K)
  for (s in 1:20) {
    A <- as.matrix(read.csv(sprintf("/tmp/m4s/diag/%s/rep%02d.csv", tipo, s), header=FALSE))
    r <- try({
      Xtr <- A[1:T,-1,drop=FALSE]; Xte <- A[(T+1):(T+H),-1,drop=FALSE]
      colnames(Xtr) <- colnames(Xte) <- paste0("x",1:K)
      f <- Arima(ts(A[1:T,1], frequency=S), order=ord, seasonal=list(order=sord,period=S),
                 xreg=Xtr, include.mean=FALSE, method="ML")
      list(f=as.numeric(forecast(f, xreg=Xte, h=H)$mean), b=as.numeric(tail(coef(f),K)))
    }, silent=TRUE)
    if (!inherits(r,"try-error")) { R[s,1:H] <- r$f; R[s,(H+1):(H+K)] <- r$b }
  }
  write.table(R, sprintf("/tmp/m4s/diag/R_%s.csv", tipo), sep=",", row.names=FALSE, col.names=FALSE)
}
cat("bracos R ok\n")
