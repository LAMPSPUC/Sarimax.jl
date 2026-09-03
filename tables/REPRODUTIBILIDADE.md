# Reprodutibilidade sob a tag `v1.0.0` — respostas medidas

Pergunta do Luiz: *"Rodando o script com todos os argumentos explícitos, sob a tag
`v1.0.0`, ele reproduz os números arquivados?"* Respondida por **sonda**, em processo
único, comparando **sMAPE (6 casas) e ordem selecionada**. Nunca por leitura de diff.

## A — Campanha `stable` (hourly, quarterly): **SIM**

| braço | commit da campanha | sonda sob v1.0.0 |
|---|---|---:|
| hourly / mse | `dev@fc2c482+PR26` | **12/12** |
| hourly / stable | `dev@fc2c482+PR26` | **30/30** |
| quarterly / mse | `dev@fc2c482+PR26` | **30/30** |
| quarterly / stable | `dev@fc2c482+PR26` | **30/30** |

102 séries, 102 reproduzem. Os dados ficam onde estão; `sarimax_commit` declara o commit
de produção e o `REPRODUCE.md` declara a `v1.0.0` como versão de publicação.

## B — Campanha 2×2×2×2, células com fator ATIVO: **NÃO, E SEI POR QUÊ**

Desenho com **controle**, para separar duas causas possíveis de divergência: efeito de
contenção (a campanha rodou `maxTimeSeconds = 120` por ajuste, com 5 workers) contra
mudança de comportamento do código.

| célula | stationary | invertible | warm start dispara? | 4e8cf11 (controle) | v1.0.0 |
|---|---|---|---|---:|---:|
| (F,F) `gff` | false | false | **não** | 10/10 | **10/10** |
| (T,F) `gs` | true | false | sim | 10/10 | **9/10** |
| (F,T) `gi` | false | true | sim | 10/10 | **9/10** |
| (T,T) `gsi` | true | true | sim | 10/10 | **9/10** |

**As quatro reproduzem 10/10 sob o commit da campanha em processo único** ⇒ contenção não é
a causa. Sob a `v1.0.0`, a única célula que **não** passa pelo caminho
`warmStartFromBox && (stationary || invertible)` reproduz integralmente; as três que passam
perdem exatamente uma série cada.

**É a mesma série nas três: sid 33468.** E sob a `v1.0.0` as três convergem para a **mesma**
ordem `(2,0,0,2,1,0)` e o **mesmo** sMAPE `27.005694` — que é exatamente o que a célula
`(F,F)` produz para essa série. Ou seja: com a semente de ε na escala correta, o warm start
deixa de perturbar a busca, e as células restritas passam a pousar na mesma solução da
irrestrita.

**Causa: o PR #26**, que corrigiu `applyWarmStart!` para semear os resíduos na escala interna
do modelo. Antes, a semente crua era um chute grande que empurrava a busca para outra ordem.

Nota, sem exagerar o alcance: nesta série o resultado arquivado (semente defeituosa) tem
sMAPE **melhor** — 25,33 contra 27,01. É o mesmo fenômeno de "chute grande às vezes acha
ponto melhor" que já foi medido e **retratado** como atrator degenerado. Uma série não
sustenta afirmação.

## C — Campanha 2×2×2×2, células com fator = 1: **NÃO SEI / IRREPRODUZÍVEL**

As quatro células `fator = 1` foram produzidas com o gancho de ambiente
`SARIMAX_PROBE_FATOR1`, que era um **patch aplicado apenas dentro do worktree `wt_probe`**.
Esse worktree foi destruído por uma limpeza do `/tmp`. Verificado: o gancho **não existe**
nem em `4e8cf11` nem na `v1.0.0`.

**O código que produziu essas quatro células não existe mais em lugar nenhum.** Elas são
comparáveis entre si e contra as células fator-ativo da mesma campanha, mas **não são
re-rodáveis**. O `Manifest.toml` daquela campanha também se perdeu na mesma limpeza.
Nenhum manifesto foi regenerado para ocupar o lugar do original.

## D — Achado de método: a lista de argumentos explícitos NÃO é portável entre commits

A sonda falhou com `MethodError` nas dez primeiras séries porque passava `exogDynamics`,
que **não existe** em `4e8cf11` — foi acrescentado depois (`9607556`). A regra "declare
todos os argumentos" continua correta, mas um script de replicação que rode contra mais de
um commit precisa ser **condicional à versão**. Está assim na sonda.

## E — Um erro meu que a disciplina de argumentos explícitos pegou

Na primeira passada a sonda deu 10/12 em `hourly/mse`, com duas séries divergindo em ordem.
A causa era minha: eu havia escrito `stationarityMargin = 1e-2` quando o
`DEFAULT_DOMAIN_MARGIN` é **1e-6** — confundi com o `rootMargin`, que é 1e-2. Corrigido o
valor, 12/12.

Registro porque é o argumento a favor da regra: **se as margens tivessem ficado implícitas,
a sonda teria dado 12/12 de graça** e o script publicado dependeria de um default que já
mudou várias vezes.
