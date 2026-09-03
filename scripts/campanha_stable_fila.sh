#!/bin/zsh
# Fila UNICA e sequencial. Conclusao por CONTAGEM DE LINHA, nunca por ausencia de processo.
# Ordem: do mais barato ao mais caro, para haver resultado cedo.
L=/Users/davi/dev/campanha_stable/fila.log
d(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NPROC=5
roda() {
  local freq=$1 S=$2 N=$3 braco=$4
  local pref=/Users/davi/dev/campanha_stable/${freq}_${braco}
  local feito=$(cat ${pref}_*.csv 2>/dev/null | grep -vc '^commit,')
  if [ "${feito:-0}" -ge $((N-2)) ]; then d "$freq/$braco ja completo ($feito) — pulando"; return; fi
  d "INICIANDO $freq/$braco (tinha ${feito:-0} de $N)"
  local por=$(( (N + NPROC - 1) / NPROC ))
  for i in $(seq 1 $NPROC); do
    local a=$(( (i-1)*por + 1 )); local b=$(( i*por ))
    ARM_FREQ=$freq ARM_S=$S ARM_BRACO=$braco ARM_OUT=${pref}_${i}.csv ARM_A=$a ARM_B=$b \
      julia --project=/Users/davi/dev/env_fix /Users/davi/dev/campanha_stable/runner.jl \
      > ${pref}_${i}.log 2>&1 &
  done
  wait
  local m=$(cat ${pref}_*.csv 2>/dev/null | grep -vc '^commit,')
  if [ "${m:-0}" -ge $((N-2)) ]; then d "$freq/$braco COMPLETO: $m/$N"
  else d "$freq/$braco INCOMPLETO: $m/$N — NAO avanco, corrigir antes"; exit 1; fi
}
d "campanha iniciada, $NPROC processos por braco, sem teto de tempo, mem_percent=1000"
roda hourly    24    414 mse
roda hourly    24    414 stable
roda quarterly  4  24000 mse
roda quarterly  4  24000 stable
roda daily      1   4226 mse
roda daily      1   4226 stable
d "CAMPANHA STABLE COMPLETA — as tres frequencias"
