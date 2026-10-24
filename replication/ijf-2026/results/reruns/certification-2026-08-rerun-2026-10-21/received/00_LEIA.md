# Rerun de replicação do lote SCIP — 21/10/2026

Começe por `RESULTADO_RERUN_21-10.md` (veredito, achados, medição complementar) e
`PREREG_RERUN_21-10.md` (expectativa escrita antes de rodar). `COMPARACAO_RERUN_vs_AGOSTO.csv` tem
uma linha por caso (46) com agosto e hoje lado a lado. `tabela_rerun_21-10.csv` tem os 43 casos
terminados, 35 colunas.

Por instância: `instancias/caso_<id>.csv` (série y e DGP no cabeçalho), `saida/caso_<id>.csv`
(Ipopt: objetivo, status, tempo, coeficientes; SCIP: objetivo, dual, status, status bruto, nós,
tempo, coeficientes, ‖ε‖²; gap certificado; carimbo de ambiente), `logs/caso_<id>.log` (log
completo do SCIP, verblevel padrão). Para 34/35/38: `*_clock900.*` é a rodada da fila (900 s) e
`caso_<id>.*` é a complementar (3600 s); ambas mortas.

Arnês: `scip_rerun.jl` = snapshot de 22/08 14:35 UTC + `diff_vs_snapshot_22-08_1435.patch` (só
gravação; modelo intocado). `roda_caso.sh` = um caso por processo, nice 10, relógio externo.
`Manifest_21-10.toml` tem sha256 igual ao Manifest do pacote anterior.
