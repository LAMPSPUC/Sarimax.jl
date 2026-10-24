# Gate das 1.500 séries (item g) — pacote para rodar fora da máquina do Davi (22/10/2026)

Objetivo: regenerar os brutos por série das duas rodadas do gate de 21/08 (`:penalized` e `:innovations`,
mesmas 1.500 séries da M4 mensal), que se perderam com /tmp/m4s. Agregado de agosto:
`referencia/agosto_gate_1500_agregado.txt`.

## Conteúdo

- `Sarimax.jl/` — código do commit f90e554 (git archive, sem .git) com o `Manifest.toml` de agosto
  (sha256 05f0db2d…; não é versionado no repo). O commit não está em nenhum branch do GitHub; o bundle traz
  só a tag `regen/g-f90e554-22-10`. Para conferir o hash:
  `git init x && git -C x fetch ../Sarimax_f90e554.bundle 'refs/tags/*:refs/tags/*' && git -C x rev-parse regen/g-f90e554-22-10^{commit}`
  deve dar f90e55445089dc1f82b87a96c8b88ff4165aacbb.
- `SHA256SUMS` — confira de dentro desta pasta com `shasum -a 256 -c SHA256SUMS`.
- `scripts/m4_gate1500.jl` — runner de 21/08. Só mudam a pasta dos dados (ENV `M4_DATA`) e o carimbo, que
  passa a ser o sha256 de `src/models/sarima.jl` (deve imprimir 833072403a766069). Cálculo, sorteio
  (`Random.seed!(20260814); sort(randperm(48000)[1:1500])`) e configuração do `auto` estão inalterados.
- `scripts/le_gate.jl` — o leitor de agosto que produz a tabela agregada.
- `scripts/valida_inov_vs_owa.py` — compara o `:innovations` série a série com `referencia/owa_pareado.csv`
  (ordem e sMAPE/MASE a 6 casas). O gate `:innovations` e o censo de 48 mil têm a mesma acurácia por série
  em agosto; a média de owa_pareado.csv nos 1.500 sids é 13.9395 / 0.9471, igual ao agregado.
- `referencia/gate1500_sids.txt` — os 1.500 sids sorteados; `referencia/M4_sha256.txt` — sha256 dos
  Monthly-train.csv e Monthly-test.csv usados (cópia do ForecastTester).

## Requisitos

Julia 1.12.3 (é a versão do Manifest). R não é usado pelo gate, mas o Manifest inclui RCall, que
precisa achar R ao instanciar. **Máquina livre:** cada candidato do `auto` tem limite de 10 s de parede
dentro do pacote; com a máquina carregada, ajustes estouram esse limite e a ordem escolhida muda.
Rode um braço por vez, sem outras cargas. Em agosto, com a máquina livre: 0,85 s/série no `:penalized` e
0,61 s/série no `:innovations` (≈ 37 min no total).

## Passos

```
cd gate_g_1500_para_luiz_22-10
export M4_DATA=/caminho/para/a/pasta/com/Monthly-train.csv   # M4 mensal, conferir com referencia/M4_sha256.txt
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
julia --project=Sarimax.jl -e 'using Pkg; Pkg.instantiate()'
M4_INIT=penalized   julia --project=Sarimax.jl scripts/m4_gate1500.jl aleatorio 1500 m1500_pen.csv  > log_pen.txt 2>&1
M4_INIT=innovations julia --project=Sarimax.jl scripts/m4_gate1500.jl aleatorio 1500 m1500_inov.csv > log_inov.txt 2>&1
grep -c TIME_LIMIT log_pen.txt log_inov.txt
julia scripts/le_gate.jl m1500_pen.csv m1500_inov.csv
python3 scripts/valida_inov_vs_owa.py m1500_inov.csv referencia/owa_pareado.csv valida_inov.csv
```

## Como ler

- `:innovations`: a reprodução vale série a série; o esperado é 1500/1500 iguais. Série divergente deve ser
  rodada de novo isolada, com o processo já aquecido (não como primeira série), antes de concluir.
- `:penalized`: não há referência por série; compara-se o agregado (13.9000 / 0.9528 / teto 47,4 % /
  1500 OK). Diferença com TIME_LIMIT no log = "não reproduzido sob carga".
- As colunas wall, build e solve dependem da máquina e não se reproduzem. O −28 % de tempo de agosto não é
  pareado (o `:innovations` rodou junto com o censo).
