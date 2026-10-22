#!/bin/zsh
# Um caso por processo, nice -n 10, stdout+stderr no log do caso, relogio EXTERNO de 900 s
# (como scip_wd.sh de 23/08: limits/time=600 do SCIP foi medido nao respeitado no caso 34).
caso=$1; lim=${2:-900}
RR=/Users/davi/dev/paper/rerun_scip_21-10
J=/Users/davi/.julia/juliaup/julia-1.12.3+0.aarch64.apple.darwin14/bin/julia
echo "inicio $(date '+%F %T') caso=$caso" >> $RR/logs/fila.log
nice -n 10 $J --project=/Users/davi/dev/Sarimax.jl $RR/scip_rerun.jl $caso $caso $RR/saida/caso_$caso.csv > $RR/logs/caso_$caso.log 2>&1 &
pid=$!
( sleep $lim; if kill -0 $pid 2>/dev/null; then
    echo "MORTO POR RELOGIO EXTERNO apos ${lim}s" >> $RR/logs/caso_$caso.log
    kill -9 $pid 2>/dev/null; fi ) &
wd=$!
wait $pid 2>/dev/null; rc=$?
kill $wd 2>/dev/null
echo "fim    $(date '+%F %T') caso=$caso rc=$rc" >> $RR/logs/fila.log
