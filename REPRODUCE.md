# Material de replicação — máquina `gavea`

Pacote de replicação do paper submetido ao *International Journal of Forecasting* sobre o
`Sarimax.jl`. Segue o padrão pedido pelo Luiz em 28/08 (msg 101302), o mesmo das máquinas
`icarai` e `recreio`.

---

## 1. As duas identidades de versão

Elas são **diferentes** e nunca devem ser confundidas.

| | valor |
|---|---|
| **Versão junto da qual este material é PUBLICADO** | **`v1.0.0`** (tag `a1ebc0d`) |
| **Commit sob o qual cada resultado foi PRODUZIDO** | varia por campanha — coluna `commit` de cada linha de CSV |

Nenhum resultado histórico é descrito como "produzido sob v1.0.0" se não foi. A tabela da
seção 3 diz, campanha a campanha, qual commit produziu o quê.

## 2. Máquina e ambiente

- **Hardware:** Apple M4, 10 núcleos, 24 GB de RAM, macOS (Darwin 24.6.0)
- **Julia 1.12.3**
- **Ambiente fixado em `env/Project.toml` e `env/Manifest.toml`**, resolvido contra a tag
  `v1.0.0`: MathOptInterface 1.53.0, Ipopt 1.11.0, JuMP 1.31.2, MUMPS_seq_jll 5.4.1,
  Ipopt_jll 300.1400.400+0, CSV 0.10.17, TimeSeries 0.24.2
- `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `BLAS.set_num_threads(1)` em todo runner

## 3. Inventário

### 3.1 Alimenta tabela do manuscrito

| campanha | n | commit de produção | conteúdo |
|---|---:|---|---|
| M4 `stable` hourly, dois braços | 414 + 414 | `dev@fc2c482+PR26` | `mse` vs `stable` α=0,5 |
| M4 `stable` quarterly, dois braços | 24.000 + 24.000 | `dev@fc2c482+PR26` | idem |
| M4 `stable` daily, dois braços | 4.227 + 3.998 | **`v1.0.0@a1ebc0d`** | idem, com censura declarada |
| M4 `stable` monthly | 5.000 + 5.000 + 923 | `aa68d57` | `mse`, α=0,9, α=0,5 (α=0,5 PARCIAL) |
| Campanha 2×2×2×2 | 8 × 48.000 | `4e8cf11` | `stationary` × `invertible` × fator log-det |
| `requireTermsWhenOverDifferenced` | 414 + 24.000 + 4.227 | `v1.0.0@a1ebc0d` | braço `true`; o `false` são as campanhas acima |

### 3.2 Diagnóstico que sustenta afirmação em prosa

| item | n | commit | o que sustenta |
|---|---:|---|---|
| Curva de custo `stable`/`mse` por T | 50 buscas | `dev@fc2c482+PR26` | spread de 114× no mesmo T = 4.315 |
| Inércia do `mumps_mem_percent` | 7 pares | `4e8cf11` | objetivo idêntico em 17 dígitos |
| **Atalho de busca — REPROVADO** | 3 × 359 | `4e8cf11` | ordem do `mse` + refit `stable`: reprovou no portão pré-registrado |
| **Warm start — achado RETRATADO** | 7 + 5 séries | `4e8cf11` / `v1.0.0` | atrator degenerado no canto da caixa; eu mesmo derrubei |
| Sondas de reprodutibilidade | 102 + 80 séries | `v1.0.0` e `4e8cf11` | ver `tables/REPRODUTIBILIDADE.md` |
| Reprodutor do crash do MUMPS | 2 séries | `v1.0.0` | vítima × culpado |

Os dois itens em **negrito** entram rotulados pelo que são: **um resultado reprovado no seu
próprio portão** e **um achado retratado**. Não são apresentados como positivos nem omitidos.

## 4. A pergunta central, respondida por MEDIÇÃO

> *Rodando o script com todos os argumentos explícitos, sob a tag `v1.0.0`, ele reproduz os
> números arquivados?*

Sonda: amostra semeada (`Random.seed!` fixo), reajuste em **processo único**, comparação de
sMAPE (6 casas) e ordem selecionada. Script em `scripts/sonda_v100.jl`.

- **SIM** — campanha `stable` hourly e quarterly: **102 de 102** séries reproduzem.
- **NÃO, E SEI POR QUÊ** — campanha 2×2×2×2, células com fator ativo: a célula que **não**
  passa pelo caminho `warmStartFromBox && (stationary || invertible)` reproduz 10/10; as três
  que passam perdem **a mesma** série (sid 33468). Causa: o PR #26 corrigiu a escala da
  semente de ε. Controle em `4e8cf11` reproduz 10/10 nas quatro, o que **elimina contenção**
  como explicação. Detalhe em `tables/REPRODUTIBILIDADE.md`.
- **NÃO SEI / IRREPRODUZÍVEL** — ver seção 5.

## 5. NÃO SEI — declarados, sem chute e sem regeneração

1. **As quatro células `fator = 1` do 2×2×2×2 são irreproduzíveis.** Foram produzidas com o
   gancho de ambiente `SARIMAX_PROBE_FATOR1`, que era um **patch aplicado apenas dentro do
   worktree `wt_probe`**, destruído por uma limpeza do `/tmp`. Verificado: o gancho **não
   existe** em `4e8cf11` nem em `v1.0.0`. **O código que as produziu não existe mais em lugar
   nenhum.** São comparáveis entre si e contra as células fator-ativo da mesma campanha, mas
   não são re-rodáveis.
2. **O `Manifest.toml` da campanha 2×2×2×2 se perdeu na mesma limpeza.** As oito células
   rodaram no mesmo ambiente e seguem comparáveis entre si, mas **não são re-rodáveis bit a
   bit**. **Nenhum manifesto foi regenerado para ocupar o lugar do original** — um manifesto
   de hoje seria um ambiente diferente com aparência de original.
3. **A configuração exata do braço `stable` monthly** (`aa68d57`) está no pré-registro
   `PREREG_STABLE_M4_24-08.md`, mas a linha de comando não foi arquivada. O que está
   declarado é o pré-registro, não a invocação.

## 6. Colunas de validade

Além da métrica principal, por braço: **taxa de erro**, **taxa de status não-`LOCALLY_SOLVED`**,
e, onde houver, **taxa de censura por motivo**.

### Campanha `stable`

| braço | erros | não-`LOCALLY_SOLVED` | censura |
|---|---:|---:|---|
| hourly / mse | 0 | 0 | — |
| hourly / stable | 0 | 0 | — |
| quarterly / mse | 0 | 0 | — |
| quarterly / stable | 0 | 56 (0,23%) | — |
| daily / mse | 0 | 0 | 0 |
| daily / stable | 5 | — | **229 (5,4%)** |

Censura do `daily/stable` por motivo: **198** `MUMPS_ABORT_WORKER_MORREU`, **19**
`MUMPS_DEADLOCK`, **8** `TETO_4H`, **4** `TETO_RETROATIVO_32H`. Lista completa de sids em
`censurados.csv`.

### Duas populações de censura, com semânticas OPOSTAS

- `MUMPS_ABORT_WORKER_MORREU` — o sid registrado é o **culpado**: o processo morreu durante
  ele. Gatilho sempre em série longa.
- `MUMPS_DEADLOCK` — o sid registrado é a **vítima**. O culpado é o sid da linha com status
  `ERRO` imediatamente anterior no mesmo shard. Correspondência verificada **um-para-um**:
  cinco linhas `ERRO`, cinco travamentos, sempre no sid seguinte.

Sem essa distinção, um leitor concluiria que séries curtas quebram o solver. Elas não
quebram — duas delas, com T = 994 e T = 3.317, **completam em 45 s e 15 s** quando rodadas
isoladas.

## 7. Tetos de tempo — declarados, com o histórico

Campanha com teto é **reproduzível só até escalonamento**. Portanto:

- **Nenhum teto** em hourly, quarterly, monthly, 2×2×2×2 e nos braços `mse`.
- **`daily/stable`**: rodou primeiro **sem teto**. Após 32 h com quatro workers parados em
  quatro séries e **zero** progresso, foi imposto um **teto retroativo** — 4 sids nomeados,
  motivo `TETO_RETROATIVO_32H`. Em seguida um **teto prospectivo de 2 h**, que cortou 14
  séries; como isso excedeu o critério fixado de ~10, o teto foi elevado **uma única vez**
  para **4 h** e as 14 voltaram à fila. Motivo final: `TETO_4H`, 8 séries.
- **Número de workers faz parte da configuração**: 5 workers no `daily/stable`, 3 no
  `requireTerms`, 5 nas demais.

## 8. Divergências entre script e legenda de tabela

Uma, e é a mais importante deste pacote. Está em e-mail separado ao Luiz (msg 101311) e
resumida aqui:

**No commit `aa68d57`, o das campanhas M4 de weekly/daily/hourly/yearly/quarterly, a
configuração declarada na legenda da Tabela 6 não reproduz a tabela.** A legenda diz
`:innovations`; naquele commit a guarda de objetivo lança para todo objetivo que não seja
`mse`, e o `huber` é o único que **degrada em silêncio** — `huberFallback` em **40/40**
séries weekly, devolvendo o resultado do `mse` bit a bit. Rodada como descrita, três colunas
lançariam e a do `huber` seria cópia da coluna MSE.

Corrigido depois: em `4e8cf11` e na `v1.0.0` a guarda enumera os nove objetivos e nunca
dispara.

## 9. Achado de método: a lista de argumentos explícitos NÃO é portável entre commits

A sonda falhou com `MethodError` porque passava `exogDynamics`, que **não existe** em
`aa68d57` nem em `4e8cf11` — foi acrescentado em `9607556`. A regra "declare todos os
argumentos" continua correta, mas um script que rode contra mais de um commit precisa ser
**condicional à versão**. Implementado assim em `scripts/sonda_2x2x2x2.jl`.

## 10. Onde estão os dados

**Os brutos NÃO estão nesta branch.** Vão em zip, com o `SHA256SUMS` **dentro do próprio
zip**, entregues por drive e depois ao Zenodo. Esta branch contém apenas `scripts/`, `env/`,
`tables/` e este arquivo.
