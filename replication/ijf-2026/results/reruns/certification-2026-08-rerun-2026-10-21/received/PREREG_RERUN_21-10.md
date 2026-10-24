# Pré-registro — rerun de replicação do lote SCIP (lotes 1 e 2), 21/10/2026

**Carimbado antes de lançar a fila.** Autorizado pelo Davi em 21/10 ("Pode rodar, fazer os dois").

- Arnês: `scip_rerun.jl`, derivado de `scip_prog_lote2_22-08_1435.jl`. Modelo (monta, B, bound OBBT,
  warm start, `limits/time = 600`) **idêntico**; só a gravação muda. Prova: `diff_vs_snapshot_22-08_1435.patch`.
  `display/verblevel` volta ao default do SCIP (log por caso em `logs/`); não altera a busca.
  `max_wall_time = 600` no Ipopt vale agora para os 46 casos (em agosto, só a partir de 22/08 14:35 UTC);
  o Ipopt levou < 1 s em todos os casos registrados, então é inerte.
- Ambiente: `~/dev/Sarimax.jl` com o Manifest de 20/08 (SCIP 8.0.0 medido, Julia 1.12.3), sem resolve.
- Execução: 4 processos (máquina compartilhada; DESSEM e Analytics avisadas e de acordo), `nice -n 10`,
  BLAS serial, um caso por processo, relógio externo de 900 s por caso. Fila: os ~23 casos com
  TIME_LIMIT esperado primeiro, depois os rápidos.
- **Expectativa escrita antes:** nos casos OPTIMAL, objetivo do SCIP igual ao registro de agosto a ~1e−6
  (valores recuperados: 33 = 435,5372; 36 = 928,6367; 44 = 57,536983; 45 = 53,681295; e o Ipopt do 26 =
  46,660913, do 37 = 938,4454, do 38 = 969,4430). Tempos e gaps de TIME_LIMIT **não** são comparáveis
  (dependem da carga). 34 e 35 podem não terminar de novo. **Divergência num OPTIMAL é achado a reportar,
  não alvo de ajuste.**
- Saída durável em `~/dev/paper/rerun_scip_21-10/` (nunca `/tmp`).
