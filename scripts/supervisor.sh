#!/bin/zsh
# Supervisor v2 — substitui o `for ... & wait`, que nao sobrevive ao deadlock do MUMPS.
# O worker NAO morre quando o MUMPS aborta: fica vivo com CPU congelada, preso num mutex.
# Logo: vigia CPU POR WORKER (nunca o total — um irmao ocupado mascara o travado),
# mata, CENSURA o sid marcado, e RECRIA o processo. Repetir dentro do mesmo processo e
# inutil: o mutex continua travado.
BASE=/Users/davi/dev/campanha_daily
L=$BASE/supervisor.log
CENS=$BASE/censurados.csv
ENVJ=/Users/davi/dev/env_v100
DATA=/Users/davi/dev/ForecastTester/datasets
NPROC=5; N=4226; FREQ=daily; S=1
d(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ -f $CENS ] || echo "quando,sid,braco,motivo" > $CENS
spawn() {  # $1 = shard, $2 = braco
  local i=$1 b=$2
  local por=$(( (N + NPROC - 1) / NPROC ))
  local a=$(( (i-1)*por + 1 )) bb=$(( i*por ))
  ARM_DATA=$DATA ARM_FREQ=$FREQ ARM_S=$S ARM_BRACO=$b \
  ARM_OUT=$BASE/${FREQ}_${b}_${i}.csv ARM_MARK=$BASE/.tentando_${b}_${i} \
  ARM_CENS=$CENS ARM_A=$a ARM_B=$bb \
    julia --project=$ENVJ $BASE/runner2.jl >> $BASE/${FREQ}_${b}_${i}.log 2>&1 &
  echo $!
}
roda_braco() {
  local b=$1
  local feito=$(cat $BASE/${FREQ}_${b}_*.csv 2>/dev/null | grep -vc '^commit,')
  d "INICIANDO $FREQ/$b (tinha ${feito:-0} de $N)"
  typeset -A PID CPU FROZEN
  for i in $(seq 1 $NPROC); do PID[$i]=$(spawn $i $b); CPU[$i]=""; FROZEN[$i]=0; done
  while true; do
    sleep 90
    local vivos=0
    for i in $(seq 1 $NPROC); do
      local p=${PID[$i]}
      if ! kill -0 $p 2>/dev/null; then
        # WORKER MORTO. O v1 do arnes tratava so isto; o v2 tratava so o travamento.
        # Sao os DOIS caminhos: o abort do MUMPS as vezes pendura o processo (mutex) e as
        # vezes o mata. Marcador presente = morreu no meio de uma serie => censura e recria.
        # Marcador ausente = shard terminou de verdade => nao recria.
        local sidm=$(cat $BASE/.tentando_${b}_${i} 2>/dev/null)
        if [ -n "$sidm" ]; then
          d "MORREU shard $i (pid $p) com marcador — sid $sidm censurado, recriando"
          echo "$(date '+%Y-%m-%dT%H:%M:%S'),$sidm,$b,MUMPS_ABORT_WORKER_MORREU" >> $CENS
          rm -f $BASE/.tentando_${b}_${i}
          sleep 2; PID[$i]=$(spawn $i $b); CPU[$i]=""; FROZEN[$i]=0; vivos=$((vivos+1))
        fi
        continue
      fi
      vivos=$((vivos+1))
      # TETO DE 2H POR SERIE, imposto pelo SUPERVISOR e nao pelo runner: assim nenhum
      # ajuste que termina e afetado, e nao ha duas populacoes de codigo dentro do braco.
      # Fundamento medido: a serie legitima mais cara da curva de custo fechou em ~764s
      # (13 min); 2h e ~9x isso, entao o teto nao corta o regime normal ja observado.
      local mk=$BASE/.tentando_${b}_${i}
      if [ -f $mk ]; then
        local idade=$(( $(date +%s) - $(stat -f %m $mk) ))
        if [ $idade -gt 14400 ]; then
          local sidt=$(cat $mk 2>/dev/null)
          d "TETO_4H shard $i (pid $p, ${idade}s na serie) — sid ${sidt:-NA} censurado, recriando"
          echo "$(date '+%Y-%m-%dT%H:%M:%S'),${sidt:-NA},$b,TETO_4H" >> $CENS
          kill -9 $p 2>/dev/null; rm -f $mk
          sleep 2; PID[$i]=$(spawn $i $b); CPU[$i]=""; FROZEN[$i]=0
          continue
        fi
      fi
      local t=$(ps -p $p -o time= 2>/dev/null | tr -d ' ')
      if [ "$t" = "${CPU[$i]}" ]; then FROZEN[$i]=$(( ${FROZEN[$i]} + 1 )); else FROZEN[$i]=0; fi
      CPU[$i]=$t
      if [ ${FROZEN[$i]} -ge 2 ]; then
        local sid=$(cat $BASE/.tentando_${b}_${i} 2>/dev/null)
        d "TRAVADO shard $i (pid $p, CPU congelada em $t) — sid ${sid:-DESCONHECIDO} censurado, recriando"
        echo "$(date '+%Y-%m-%dT%H:%M:%S'),${sid:-NA},$b,MUMPS_DEADLOCK" >> $CENS
        kill -9 $p 2>/dev/null; rm -f $BASE/.tentando_${b}_${i}
        sleep 2; PID[$i]=$(spawn $i $b); CPU[$i]=""; FROZEN[$i]=0
      fi
    done
    [ $vivos -eq 0 ] && break
  done
  local m=$(cat $BASE/${FREQ}_${b}_*.csv 2>/dev/null | grep -vc '^commit,')
  local c=$(grep -c ",$b,MUMPS_DEADLOCK$" $CENS 2>/dev/null)
  local alvo=$(( N - ${c:-0} ))
  if [ "${m:-0}" -ge $((alvo-2)) ]; then
    d "$FREQ/$b FECHADO: $m linhas, ${c:-0} censuradas (alvo $N, efetivo $alvo)"
  else
    d "$FREQ/$b INCOMPLETO: $m de $alvo (N=$N menos ${c:-0} censuradas) — NAO avanco"
    exit 1
  fi
}
d "supervisor v2 iniciado — tag v1.0.0, $NPROC workers, watchdog por CPU congelada por worker"
roda_braco mse
roda_braco stable
d "CAMPANHA DAILY COMPLETA sob a tag"
