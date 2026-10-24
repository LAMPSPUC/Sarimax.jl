# Rerun de replicação do lote SCIP (21/10/2026) — resultado vs. registro de agosto

Pré-registro: `PREREG_RERUN_21-10.md`. Arnês: `scip_rerun.jl` (modelo idêntico ao snapshot
`scip_prog_lote2_22-08_1435.jl`; diff em `diff_vs_snapshot_22-08_1435.patch`, só gravação).
Execução: 21/10 19:20:23 → 20:35:20 (hora de Brasília; fim = última linha `fim` de `logs/fila.log`,
caso 36), 4 processos `nice -n 10` em paralelo, máquina compartilhada, relógio externo de 900 s por
caso (`roda_caso.sh`). Caso 1 rodou antes como smoke (carimbo 19:19:38), mesmo arnês.

Carga medida (loadavg 1 min no carimbo de cada caso, 46 casos): mín 6,44, máx 12,04. Tempos de
parede dos OPTIMAL não são comparáveis aos de agosto (ex.: caso 36, 469 s hoje vs 367 s em agosto).

Ambiente: `~/dev/Sarimax.jl/Manifest.toml` sha256 `05f0db2d…4dcf`, idêntico ao Manifest dentro do
zip enviado ao Luiz às 19:17 (medido com `unzip -p | shasum`). Julia 1.12.3, SCIP 8.0.0 (versão
impressa pelo arnês). O arnês não carrega o pacote Sarimax (nenhum `using Sarimax`); o `git_head`
do carimbo (14dd581, merge do PR #33, árvore limpa) é metadado do ambiente, não do código que rodou.
Em agosto o HEAD era 60e45cc.

Nota sobre `logs/fila.log`: a linha `FILA COMPLETA` traz 19:20:23 porque o `date` expandiu no
lançamento; a hora real de término é a da última linha `fim` (20:35:20). Log não editado.

**Regime do relógio externo.** Em agosto, o lançamento original do lote 2 (22/08 10:14 UTC) não tinha
relógio externo; o relógio de 900 s (`scip_wd.sh`) existiu só nas relançadas de 34/35. Hoje o relógio
de 900 s foi aplicado aos 46 casos. Isso muda a leitura do caso 38 (abaixo).

Arquivos: `tabela_rerun_21-10.csv` (43 linhas × 35 colunas), `COMPARACAO_RERUN_vs_AGOSTO.csv`
(46 linhas), `saida/caso_<id>.csv`, `instancias/caso_<id>.csv` (y + DGP), `logs/caso_<id>.log`
(log completo do SCIP), `logs/fila.log`.

## Veredito por grupo

| grupo | n | resultado |
|---|---|---|
| OPTIMAL em agosto | 23 | 23/23 OPTIMAL de novo. gap certificado igual a 3 algarismos em 23/23. Objetivo do SCIP igual nos 4 casos com valor registrado (33, 36, 44, 45; \|Δ\| ≤ 6e−6, registro de agosto tinha 4 casas). Ipopt−SCIP igual nos 16 casos do lote 1 com valor registrado. |
| TIME_LIMIT em agosto | 20 | 19/20 TIME_LIMIT de novo; 1 (caso 38) morto pelo relógio externo de 900 s (ver abaixo). gap e tempo não comparáveis (pré-registro). Incumbente e objetivo do Ipopt iguais nos 2 casos com valor registrado (26, 37). |
| mortos em agosto (34, 35) | 2 | mortos de novo aos 900 s. |

Nenhum OPTIMAL divergiu. Nenhuma troca de status OPTIMAL↔TIME_LIMIT.

## Achados

1. **Caso 38 (E, T=960)**: agosto TIME_LIMIT em 600,9 s (dual 401,19, gap 0,586), **sem relógio
   externo**. Hoje, sob relógio de 900 s, SCIP imprimiu a última linha de log aos 355 s (100 nós, gap
   interno 3,01 %) e não respondeu mais; morto aos 900 s. Não é o mesmo regime de agosto: não se sabe
   se, sem o relógio, teria terminado em ~600 s como em agosto. Mesmo padrão do caso 34 em agosto
   (`limits/time=600` não respeitado). Medido, não explicado.
2. **Caso 34 (E, T=480)**: última linha aos 217 s (300 nós, gap interno 0,53 %, incumbente 455,2423 =
   Ipopt), depois silêncio até a morte aos 900 s. Igual a agosto (duas tentativas).
3. **Caso 35 (E, T=480)**: Ipopt OK em 1,6 s (485,4985); SCIP não imprimiu nenhuma linha da tabela
   em ~898 s. Igual a agosto.
4. **Caso 37 (E, T=960)**: TIME_LIMIT com 0 nós e dual = −1e+20 (sentinela), idêntico a agosto. SCIP
   gasta os 600 s antes de fechar o nó raiz. Logo o "gap" 1,07e+17 não é gap.
5. Casos 28–30 (M4 reais): `piso_ok = false` de novo, como em agosto.
6. Relógio externo de 900 s: casos 34/35/38 travaram em `limits/time=600` ignorado. Explicação
   não medida; candidata (sub-NLP do Ipopt dentro do SCIP sem limite próprio) é hipótese.

## O que hoje existe e em agosto não existia

Por instância: série y e DGP (`instancias/`), coeficientes e status do Ipopt, nós, status bruto,
coeficientes e ‖ε‖² do SCIP, log completo do SCIP, carimbo (Dates.now, host, Julia, sha256 do
Manifest, HEAD, loadavg). Itens a–e do pedido do Luiz, para os 43 casos que terminaram.

## Medição complementar (fora do pré-registro): 34, 35, 38 com relógio de 3600 s

Lançados 21/10 20:40:26, 3 processos `nice -n 10`, para responder se o SCIP retorna ou trava nesses
casos. Logs de 900 s preservados em `logs/caso_<id>_clock900.log`.

**Resultado (21/10 20:40 → 21:40): os três foram mortos aos 3600 s.** Nenhum terminou; `saida/caso_<id>.csv`
vazio. O SCIP não devolveu controle em 1 h apesar de `limits/time=600`:

| caso | T | Ipopt | última linha de log do SCIP | depois |
|---|---|---|---|---|
| 34 | 480 | 455,2423 em 1,6 s | 218 s, 300 nós, incumbente 455,2423, dual 452,8227, gap interno 0,53 % | silêncio por ~56 min |
| 35 | 480 | 485,4985 em 1,5 s | nenhuma linha da tabela em ~60 min | — |
| 38 | 960 | 969,4430 em 1,6 s | 346 s, 100 nós, incumbente 969,4430, dual 941,1256, gap interno 3,01 % | silêncio por ~54 min |

Repetição exata do padrão de 900 s (mesmas últimas linhas, a ~1 s). Em 34 e 38 o incumbente já
era o do Ipopt e o dual estava a 0,5 % / 3 % quando o SCIP travou; não há certificado porque o
processo não retorna. Causa não medida. Hipótese (não verificada): chamada a sub-NLP (heurística
`subnlp`, visível no log de 38) sem limite próprio de tempo. Em agosto, 38 terminou TIME_LIMIT em
600,9 s sem relógio externo; hoje travou duas vezes. Também não medido por quê.

## Correção 22/10/2026 — log do SCIP em buffer (anula três frases acima)

**O log do SCIP sai em buffer de bloco quando o stdout vai para arquivo.** Medido em 22/10: no caso 34
rodado sozinho, aos 4 e aos 7 minutos a última linha visível no log era a dos 88,1 s; ao terminar, o log
tinha a linha dos 418 s e o resumo final. Num processo morto com `kill -9`, o buffer se perde.

Prova nos logs de 21/10: a parte do SCIP (depois de `scip...`) tem exatamente o mesmo tamanho nas duas
rodadas mortas de cada caso — 8.531 bytes no caso 34 (900 s e 3600 s) e 8.960 bytes no caso 38.
O corte é a fronteira do buffer, não o ponto em que o SCIP estava.

Ficam anuladas, aqui e no e-mail ao Luiz de 22/10 07:55:
- "última linha de log aos 218 s (34) / 346 s (38) e depois silêncio por quase 1 h";
- "o padrão repete ao segundo";
- a hipótese da heurística sub-NLP, que se apoiava nessas linhas.

Continua verdadeiro: na fila de 21/10 (3–4 SCIP simultâneos) e na complementar (34/35/38 juntos), os
três processos não retornaram em 900 s nem em 3600 s, com `limits/time = 600`.

**Caso 34 rodado sozinho em 22/10 (mesmo arnês, mesmos parâmetros, load 50–80):** terminou.
TIME_LIMIT, solving time 600,01 s, 182 nós, incumbente 455,242254 (= Ipopt), dual 432,000151, gap 5,38 %
(gap certificado 0,0511). Arquivos: `saida/caso_34_sozinho_22-10.csv`, `logs/caso_34_sozinho_22-10.log`.
A diferença em relação às rodadas mortas não está medida.

| caso 34 | condição | resultado |
|---|---|---|
| 21/10 fila | 4 SCIP simultâneos, relógio 900 s | morto aos 900 s |
| 21/10 complementar | 34/35/38 simultâneos, relógio 3600 s | morto aos 3600 s |
| 22/10 sozinho | 1 SCIP, relógio 900 s | TIME_LIMIT 600,01 s, 182 nós, gap 5,38 % |

**Caso 35 (22/10, sozinho, arnês original):** morto aos 900 s. Amostra de pilha aos 2 min dentro da
heurística completesol, num sub-SCIP, em fatoração LU do SoPlex. A completesol roda porque o SCIP.jl 0.11.6
entrega todo warm start como solução parcial (`SCIPcreatePartialSol`). Com `heuristics/completesol/freq = -1`
(diagnóstico, parâmetro alterado, não é replicação): presolve 0,04 s, incumbente igual ao objetivo do Ipopt
(485,4985) aos 0,5 s, e o processo também foi morto aos 900 s. Detalhes em `diag_trava_22-10/NOTAS_DIAG_TRAVA_22-10.md`.
Caso 35 de novo sozinho (11:49–12:04, arnês original): amostras aos 12 e aos 14 min (720 s e 840 s) na
**mesma pilha** dos 2 min: `SCIPpresolve → heurExecCompletesol → sub-SCIP → SCIPlpSolveAndEval → SoPlex →
CLUFactor::selectPivots`. O processo passa os 15 minutos dentro da completesol no presolve; o SCIP principal
não sai do presolve e o `limits/time = 600` não interrompe. Morto aos 900 s. Esta é a causa medida para o 35.

**Caso 38 sozinho em 22/10 (12:04–12:18, arnês original):** terminou. TIME_LIMIT, solving time 600,01 s,
111 nós, incumbente 969,442960 (= Ipopt), dual 950,769961, gap 1,96 % (gap certificado 0,0193).
Agosto (5 processos, sem relógio externo): TIME_LIMIT 600,9 s, dual 401,19. Gap e dual de TIME_LIMIT
dependem da carga (pré-registro). Arquivos: `saida/caso_38_sozinho_22-10.csv`, `logs/caso_38_sozinho_22-10.log`.

Resumo dos três: **35** não retorna por causa medida (completesol no presolve); **34 e 38** retornam
normalmente em 600 s quando rodam sozinhos, e não retornaram com 3–4 SCIP simultâneos em 21/10. A causa
desta última diferença não está medida.
