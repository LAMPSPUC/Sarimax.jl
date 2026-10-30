# Inventário — pedido do Luiz de 20/10 (e-mail 121531, thread "Resultados do Sarimax")

**Lido em 21/10/2026.** Luiz confirmou os itens 2 e 3 do e-mail de 16/10 e pediu seis coisas
sobre o lote de certificação SCIP de agosto (a–f) e três registros de diagnóstico (g–i).
Este inventário diz, item a item, **o que existe, onde, e o que se perdeu** — medido, não inferido.
**Horários:** todos os carimbos `_HHMM` nos nomes de arquivo e os horários citados como "do transcript" estão em **UTC** (3 h à frente do horário de Brasília).

Este arquivo acompanha o zip `scip_agosto_reconstrucao_21-10`; os caminhos abaixo que começam
em `~/dev/` são da máquina do Davi e os arquivos correspondentes estão nas pastas do zip.

## Fatos de correio

| | |
|---|---|
| "Fechamento de 21/08" (97407) | **foi enviado** em 22/08 07:13 para [the author's address] (pasta Sent), com 5 anexos `.md` (RESULTADO_SCIP_LOTE1_21-08.md entre eles). Só resumos; nenhum dado bruto. |
| RESULTADO_SCIP_LOTE2_22-08.md | **nunca foi anexado a e-mail nenhum** (97798 e 98325 sem anexos). Existe em `~/dev/paper/`. |
| Brutos do SCIP (CSV/log) | **nunca foram enviados**. Não estão no brutos_gavea.zip do Drive (0 arquivos "scip" em 187). |

## Onde os brutos viviam — e por que não estão mais lá

Os dois lotes rodaram em `/tmp/m4s/` (arnês `scip_prog.jl`; saídas `scip_prog_s1..8.csv`,
`scip_lote1.csv`, `scip_l2_1..5.csv`, `scip_l2_1b.csv`, `scip_l2_c34/35.csv|log`). **`/tmp/m4s` não
existe mais** (a causa não foi medida; `/tmp` do macOS é volátil). A cópia parcial de 28/08 em `~/dev/resgate_tmp/` (368 entradas)
**não contém nenhum arquivo do SCIP**. O scratchpad da sessão de agosto também já não existe.

O que foi recuperado veio do **transcript da sessão** (`~/.claude/projects/.../a4d2e4b4-….jsonl`,
54 MB): código escrito via heredoc e tabelas impressas em stdout.

## Itens a–f (lote de certificação SCIP, Tabela S5 e Seção 5.6)

| item | existe? | onde / detalhe |
|---|---|---|
| a) instâncias | **sim, integralmente** | `casos()` do arnês reconstruído define as 46 instâncias (lote 1: ids 1–32, células A/B/C/D; lote 2: ids 33–46, células E/F/G) com T, ordem, modo, DGP e semente. Séries M4 reais: sids 2772, 7389, 7510 (ordem fixa (1,0,1)(0,0,1)_12). |
| b) saída do SCIP | **parcial** | O CSV persistia 17 colunas (`obj_ipopt, obj_scip, dual_scip, status, t_scip, gap_rel, gap_certificado, piso_ok`…) — **o CSV se perdeu**. Recuperado do stdout: lote 1, 32/32 linhas com status, tempo, gap certificado, Ipopt−SCIP e piso (sem primal/dual numéricos, exceto caso 26); lote 2, 12/14 linhas com status, tempo e gap, mais objetivo e dual do SCIP em 33, 36, 37, 38, 44, 45 (o dual de 37 é a sentinela −1e+20, sem limite válido); casos 34 e 35 só com o objetivo do Ipopt, mortos por relógio externo aos 900 s. **Número de nós: nunca foi gravado.** **Logs do SCIP: nunca existiram** (`display/verblevel = 0`). Status bruto: só o `termination_status` do JuMP. Tabela consolidada: `tabela_scip_lotes1-2_RECONSTRUIDA.csv` (raiz do zip) (46 linhas, coluna `fonte`). |
| c) incumbente do Ipopt | **só o objetivo** | Coeficientes do Ipopt eram calculados em memória e usados no warm start, **não gravados**. Objetivo do Ipopt em 9 casos (26, 33–38, 44, 45); os demais só via Ipopt−SCIP (lote 1). Status do Ipopt não gravado. |
| d) harness | **sim** | `harness/scip_prog_lote1_21-08_1843.jl` (como rodou o lote 1) e `scip_prog_lote2_22-08_1014.jl` / `_1435.jl` (lote 2; a segunda com `max_wall_time=600` no Ipopt), reconstruídos pela reaplicação dos patches do transcript (todas as âncoras encontradas) e **verificados por execução**: `include` com fatia vazia imprime `0 casos nesta fatia (de 32)` / `(de 46)`, SCIP 8.0.0, nos três snapshots. Lifting: `uϕΦ == ϕ*Φ`, `uθΘ == θ*Θ`, caixas ±B² com B = 1−1e−6; `|ε_t| ≤ sqrt(f_Ipopt)·(1+1e−9)`, f_Ipopt = objetivo do Ipopt no mesmo modelo (ε livres se o Ipopt falha); warm start completo (coeficientes, vetor ε, produtos, e0/z0). Mais `scip_wd.sh` (relógio externo de 900 s para 34/35). |
| e) ambiente | **sim** | Projeto `~/dev/Sarimax.jl`; `Manifest.toml` com mtime 20/08 10:49 e inalterado: SCIP.jl 0.11.6, SCIP_jll 0.2.1+0, Ipopt.jl 1.15.0, Ipopt_jll 300.1400.1902+0, JuMP 1.31.1, Julia 1.12.3. **Versão do SCIP medida hoje nesse ambiente: `SCIP_versionnumber() = 8.0.0`.** Parâmetros do SCIP: `limits/time = 600`, `display/verblevel = 0`, **nada mais alterado** (gap, absgap, feastol nos defaults). Ipopt: `print_level = 0`; a partir de 22/08 14:35 também `max_wall_time = 600`. **Paralelismo (medido nos comandos de disparo):** lote 1 = 8 processos × 4 casos, 21/08 18:43; lote 2 = 5 processos (33–35, 36–38, 39–41, 42–44, 45–46), 22/08 10:14; 34–35 relançados em processo próprio 22/08 14:35 e de novo 23/08 16:04 sob relógio externo de 900 s. BLAS serial em cada processo. O arnês não carrega o Sarimax; HEAD do pacote no reflog em 21/08: 60e45cc (irrelevante para o arnês). |
| f) mapeamento | **sim** | Quase-cancelamento: **id 26** (célula C, (1,0,1)(1,0,1)_12, Φ=0,700, Θ=−0,699, T=60): Ipopt 46,660913 / SCIP 45,314976 / dual 30,738742, TIME_LIMIT. Certificados da 5.6: **id 44** (F, cancel-regular folga 5e−2, :zeroed) 57,868044 vs 57,536983, gap 1,09e−9; **id 45** (G, inov-cancel-regular, :innovations) 54,706978 vs 53,681295, gap 1,07e−8. Grade sazonal = célula B, ids 13–24. |

### Discrepâncias entre o texto do Luiz e o registro (para ele ver, não verdicto)

1. "Incumbente 2,5% abaixo do local" na grade sazonal. Registro: **o incumbente do SCIP coincide
   com o ponto do Ipopt a ≤ 2,5e−9** (2,5 × 10⁻⁹, não 2,5%).
2. "Gaps restantes de 1,9%, 4,9% e 1,2%". Registro da célula B: melhor **1,9e−4**, mediana **~1,2e−2** (doc; nas 12 linhas recuperadas, 1,5e−2),
   pior **4,9e−1** — em percentual, 0,019%, 1,2% e 49%.
3. "T=480 certifica em 136 s / a fronteira vai até 480" (RESULTADO_LOTE2) apoia-se em **1 de 3**:
   ids 33–35 são T=480; 33 OPTIMAL em 136 s, **34 e 35 sem resultado** (mortos por relógio externo
   aos 900 s; o próprio `scip_wd.sh` registra que o 34 passou 7h40 num `limits/time` de 600 s antes
   de existir o relógio externo). Em T=960: 36 OPTIMAL, 37 TIME_LIMIT sem dual, 38 TIME_LIMIT gap 0,586.
4. Contagem: Tabela S5 "37 instâncias" **bate com A(12) + B(12) + C-sintéticas(3) + D(2) + E(6) + G(2)**.
   Se esse for o recorte da S5, ela omite as 3 séries reais da M4 (C, ids 28–30) e as 6 da célula F — mas o caso 44, que
   ancora a Seção 5.6, **é da célula F**. A célula "modo pré-amostral livre T∈{60,120}" não existe
   com esse nome; o que há em `:innovations` são D (T=60, 2) e G (T=60 e T=120).

## Itens g, h, i

| item | existe? | detalhe |
|---|---|---|
| g) PR #20, 10 sementes | **script e tabela por semente sim; séries não** | `g4.jl` (braços A/B, :penalized) e `g4p.jl` (C/D, :innovations): T=200, verdade [0,6; 0,4; 0,7; −0,4], sementes `MersenneTwister(500+r)`, r=1..10, alvo externo `arima(method="ML")` do R nos mesmos dados via Rscript. Saída por semente recuperada (10 linhas cada; medianas 0,2076 / 0,4294 / 0,0863 / 0,7984). Os `y$r.csv` se perderam; gerador determinístico. Gate pré-registrado em `~/dev/paper/GATE_TERMO_CRUZADO_20-08.md`. |
| g) 1.500 pareadas, 47,4% → 0,2% | **script reconstruído; por ajuste não** | Runner = `~/dev/m4-results/m4_runner.jl` (em disco) + parametrização `M4_INIT` + carimbo de commit (reconstruído: `scripts_g/m4_inov_gate1500_21-08_0128.jl`, âncoras encontradas); `aleatorio 1500` com `Random.seed!(20260814)`; leitor `le_gate.jl` (bootstrap, semente 20260821). Saída agregada recuperada (1273 s vs 914 s = −28%; commit f90e554). `m1500_pen.csv` / `m1500_inov.csv` **perdidos**. Do censo de 48.000 com :innovations, **a acurácia por série sobrevive em disco**: `~/dev/paper/scripts/owa_pareado.csv` (48.000 linhas: sid, T, ordem, sMAPE/MASE do Julia, do R e do Naive) e `censo_blocos.csv` (por bloco de horizonte); **só a telemetria por ajuste (iterMax etc., o "teto 200 em 0,1%") se perdeu** com `m4_inov_full.csv`. Os `~/dev/resgate_tmp/f1_*.csv` são **outro censo** (fator 1, 24/08, `m4_fator1.jl`), não este. |
| h) Tabela S2, 3 processos × 20 réplicas | **gerador, ajustador e leitor sim; CSVs por réplica não** | `diag_gera.jl` (sementes 9000+s, s=1..20, T=180, H=12, K=3, β=[2,0; −1,5; 0,8], σ_e=0,9846, DGPs erros_arima / ma_puro / armax), `diag_jl.jl` (lado Julia), `diag_R.R` (lado R: `forecast::Arima(xreg=, method="ML")`), `diag_ler.jl` e a versão RCall `diag_armax.jl` recuperados. Saída agregada por DGP recuperada (DISCRIMINADOR). `~/dev/resgate_tmp/diag/*/` existe mas **está vazio**. Gerador determinístico. |
| i) 1.099 ajustes / 21,6 → 3,3 | **scripts sim; por ajuste não — e são duas medições distintas** | (1) **1.099 ajustes pareados** = paridade de AICc na grade P×Q, as 100 primeiras da lista `rand400.txt` (a lista se perdeu; os 400 sids da amostra estão em `dados_em_disco/rand400.csv`), R via `aicq.R` (`forecast::Arima`, method CSS-ML) e `aicq.jl`; saída recuperada (k idêntico 1.099/1.099; ΔAICc +8,76 e +14,83). (2) **Distância CSS→ML 21,6 → 3,3** = `mixdist.jl` contra `rmix.csv` (loglik de ML do R, modelos mistos): **n = 14 antes e n = 12 depois — não pareado** (dois ajustes descartados pelo script); "antes/depois" = `seasonalPresamplePrior = false/true` **na mesma revisão** (16/08), não duas revisões. (3) 12,1% → 16,5% e OWA +0,0097 = `~/dev/m4-results/g_rand400.csv` e `fix_rand400.csv` (em disco, 15/08) e FRONTEIRA_FINAL.md §15–16. O texto do paper junta (1) e (2) numa frase. |

## O que dá para entregar sem rodar nada

- Arnês reconstruído (3 snapshots que rodaram) + `scip_wd.sh` + tabela consolidada de 46 linhas com proveniência.
- RESULTADO_SCIP_LOTE1/LOTE2, GATE_SCIP_21-08 (em `~/dev/paper/`).
- Manifest/Project do ambiente (`~/dev/Sarimax.jl`, mtime 20/08) + versão medida do SCIP.
- Scripts g/h/i recuperados ou reconstruídos + saídas agregadas; `rand400.csv`, `g_rand400.csv`,
  `fix_rand400.csv`, `pb*_mlrefit.jl` e CSVs (`~/dev/m4-results/`).

## O que só volta rodando de novo

Por instância do SCIP: primal/dual numéricos das 37 linhas sem objetivo, coeficientes e status do
Ipopt, nós. Por réplica/ajuste de g, h, i(1), i(2). Tudo é regenerável: sementes e código estão
íntegros. Custo de parede do lote 1 em 8 processos: ~1h (12 sazonais × 600 s distribuídas); lote 2
similar, mais os dois casos que não terminam.
