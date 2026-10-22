# Diagnóstico dos casos que não retornaram (34, 35, 38) — 22/10/2026

Tudo medido nesta data, máquina compartilhada (load 1 min entre 7 e 82 durante as rodadas).

1. **O log do SCIP sai em buffer de bloco quando o stdout vai para arquivo.** No caso 34 de hoje,
   aos 4 e aos 7 minutos a última linha visível no log era a dos 88,1 s; ao terminar, o log tinha a
   linha dos 418 s e o resumo final. Consequência: em processos mortos com `kill -9`, o que estava no
   buffer se perde, e a "última linha" visível NÃO indica onde o SCIP estava. As frases "silêncio
   depois dos 218 s" e "o padrão repete ao segundo" (relatório e e-mail de 22/10 07:55) descrevem a
   fronteira do buffer, não um travamento.
2. **Caso 34 sozinho, arnês original (`scip_rerun.jl`), 11:34–11:45:** terminou. TIME_LIMIT, raw
   SCIP_STATUS_TIMELIMIT, solving time 600,01 s, 182 nós, incumbente 455,242254 (= Ipopt), dual
   432,000151, gap 5,38 % (gap certificado 0,0511). Amostras de pilha aos 4 e 7 min: dentro de
   `SCIPsolve → SCIPsolveCIP` (branch-and-bound normal). Na fila de 21/10 (4 processos) e na
   complementar (3 processos, 34/35/38 juntos) o mesmo caso foi morto aos 900 s e aos 3600 s.
3. **Caso 35 sozinho, arnês original, 10:10–10:25:** morto aos 900 s. Amostra aos 2 min:
   `SCIPsolve → SCIPpresolve → SCIPprimalHeuristics → heurExecCompletesol → SCIPsolve (sub-SCIP) →
   SCIPsolveCIP → SCIPlpSolveAndEval → SoPlex … CLUFactor::selectPivots`. A completesol roda porque o
   SCIP.jl 0.11.6 sempre entrega o warm start como solução parcial (`SCIPcreatePartialSol`,
   `MOI_wrapper.jl:243`).
4. **Caso 35 com `heuristics/completesol/freq = -1` (diagnóstico, parâmetro alterado, não é
   replicação):** presolve 0,04 s; incumbente 485,4985 (= Ipopt) aos 0,5 s; última linha visível
   560 s (200 nós, gap 0,06 %); morto aos 900 s. Pelo item 1, a linha dos 560 s não diz onde parou.
   (O incumbente igual ao Ipopt aos 0,5 s é o que foi medido; se o warm start foi usado como tal não foi.)
5. **Caso 35 sozinho de novo, arnês original:** amostras aos 12 e 14 min na mesma pilha dos 2 min
   (presolve → completesol → sub-SCIP → LP → SoPlex selectPivots). Causa medida do não-retorno do 35.
6. **Caso 38 sozinho, arnês original:** terminou. TIME_LIMIT 600,01 s, 111 nós, gap 1,96 %.

Não medido: por que 34 e 38 terminam sozinhos e não terminaram com 3–4 SCIP simultâneos em 21/10.
Hipótese não testada: depende de quão longe a busca chega antes dos 600 s (34 de 21/10 tinha 300 nós aos
217 s; o de hoje, 100 nós aos 418 s); testável com `limits/time` maior no 34 sozinho. A hipótese "sub-NLP" do e-mail ao Luiz não
tem suporte medido.
