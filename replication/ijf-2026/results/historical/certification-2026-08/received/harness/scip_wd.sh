#!/bin/zsh
# Relogio EXTERNO. A configuracao do SCIP nao muda — o pre-registro exige uma so,
# com defaults. Mas o `limits/time` ja foi medido NAO respeitado no caso 34 (7h40
# num teto de 600s), entao o limite passa a ser imposto de fora e o caso e
# classificado como "sem certificado, morto por relogio externo".
caso=$1; lim=${2:-900}
julia --project=/Users/davi/dev/Sarimax.jl /tmp/m4s/scip_prog.jl $caso $caso \
      /tmp/m4s/scip_l2_c$caso.csv > /tmp/m4s/scip_l2_c$caso.log 2>&1 &
pid=$!
( sleep $lim; if kill -0 $pid 2>/dev/null; then
    echo "MORTO POR RELOGIO EXTERNO apos ${lim}s" >> /tmp/m4s/scip_l2_c$caso.log
    kill -9 $pid 2>/dev/null; fi ) &
wd=$!
wait $pid 2>/dev/null
kill $wd 2>/dev/null
echo "caso $caso encerrado"
