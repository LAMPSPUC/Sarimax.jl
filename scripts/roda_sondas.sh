#!/bin/zsh
# Sondas de reprodutibilidade dos bracos ja arquivados, contra a tag v1.0.0.
# Rodam em UM processo, em serie — comparam sMAPE e ordem, que sao deterministicos sob
# carga; so relogio nao seria. Por isso podem correr ao lado da campanha daily.
O=/Users/davi/dev/pacote_gavea/tables
mkdir -p $O
for spec in "hourly stable 24 30" "quarterly mse 4 30" "quarterly stable 4 30"; do
  set -- $=spec
  echo "=== sonda $1/$2 (n=$4)"
  SONDA_FREQ=$1 SONDA_BRACO=$2 SONDA_S=$3 SONDA_N=$4 \
    julia --project=/Users/davi/dev/env_v100 \
    /Users/davi/dev/pacote_gavea/scripts/sonda_v100.jl \
    > $O/sonda_$1_$2.txt 2>&1
  tail -1 $O/sonda_$1_$2.txt
done
echo "SONDAS COMPLETAS"
