#!/bin/zsh
# requireTerms = TRUE nas tres frequencias. O braco FALSE ja existe (campanha mse).
# Objetivo `mse` so — e o que o experimento do Luiz compara. 3 workers para nao afogar
# a campanha daily que roda em paralelo.
BASE=/Users/davi/dev/campanha_reqterms
L=$BASE/fila_rt.log
d(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NP=3
roda() {
  local freq=$1 S=$2 N=$3
  local feito=$(cat $BASE/${freq}_mse_*.csv 2>/dev/null | grep -vc '^commit,')
  [ "${feito:-0}" -ge $((N-2)) ] && { d "$freq ja completo ($feito)"; return; }
  d "INICIANDO $freq (tinha ${feito:-0} de $N)"
  local por=$(( (N + NP - 1) / NP ))
  for i in $(seq 1 $NP); do
    local a=$(( (i-1)*por + 1 )); local b=$(( i*por ))
    ARM_DATA=/Users/davi/dev/ForecastTester/datasets ARM_FREQ=$freq ARM_S=$S ARM_BRACO=mse \
    ARM_OUT=$BASE/${freq}_mse_${i}.csv ARM_MARK=$BASE/.t_${freq}_${i} \
    ARM_CENS=$BASE/cens_rt.csv ARM_A=$a ARM_B=$b \
      julia --project=/Users/davi/dev/env_v100 $BASE/runner_rt.jl > $BASE/${freq}_${i}.log 2>&1 &
  done
  wait
  d "$freq FECHADO: $(cat $BASE/${freq}_mse_*.csv 2>/dev/null | grep -vc '^commit,')/$N"
}
[ -f $BASE/cens_rt.csv ] || echo "quando,sid,braco,motivo" > $BASE/cens_rt.csv
d "requireTerms=TRUE — hourly, quarterly, daily; $NP workers"
roda hourly    24    414
roda quarterly  4  24000
roda daily      1  4226
d "REQUIRETERMS COMPLETO"
