#!/bin/zsh
O=/Users/davi/dev/pacote_gavea/tables
for spec in "gff 0 0" "gs 1 0" "gi 0 1" "gsi 1 1"; do
  set -- $=spec
  for amb in "4e8cf11:/Users/davi/dev/env_4e8" "v1.0.0:/Users/davi/dev/env_v100"; do
    tag=${amb%%:*}; env=${amb##*:}
    f=$O/censo_$1_${tag}.txt
    [ -f $f ] && grep -q RESULTADO $f && continue
    CEL=$1 STAT=$2 INV=$3 SONDA_N=10 TAG=$tag \
      julia --project=$env /Users/davi/dev/pacote_gavea/scripts/sonda_2x2x2x2.jl > $f 2>&1
    echo "$(grep RESULTADO $f 2>/dev/null || echo "$1/$tag: sem resultado")"
  done
done
echo "SONDAS DO CENSO COMPLETAS"
