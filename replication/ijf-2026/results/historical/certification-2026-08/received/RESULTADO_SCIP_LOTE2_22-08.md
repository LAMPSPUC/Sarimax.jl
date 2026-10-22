# SCIP lote 2 — extensão exploratória · 22/08

**Extensão declarada:** as células E/F/G foram escolhidas **à luz do parcial do
lote 1**, e isso está escrito no próprio código do arnês, não só aqui. `T = 480`
e `960` porque `T = 240` certificou; mais adversariais porque a única refutação do
lote 1 apareceu ali. **Exploração escrita é honesta; silenciosa seria o pecado.**
Leitura separada do lote 1.

## Célula F — a pergunta que o lote 1 abriu, respondida

O lote 1 refutou no quase-cancelamento **sazonal** e não no **regular**. Duas
explicações competiam: a folga do cancelamento, ou o bloco. A célula F varre a
folga nos dois blocos.

| caso | folga | status | t | Ipopt − SCIP |
|---|---|---|---:|---:|
| cancelamento **regular** | 1e−4 | **OPTIMAL** | 267 s | 0,0001% |
| cancelamento **regular** | 1e−2 | **OPTIMAL** | 12 s | 0,0001% |
| cancelamento **regular** | 5e−2 | **OPTIMAL** | 521 s | **0,5754%** |
| cancelamento **sazonal** | 1e−4 | TIME_LIMIT | 605 s | 0,0000% |
| cancelamento **sazonal** | 1e−2 | TIME_LIMIT | 600 s | 0,0000% |
| cancelamento **sazonal** | 5e−2 | TIME_LIMIT | 600 s | 0,0000% |

**A certificabilidade é propriedade do bloco sazonal, não da folga.** O regular
certifica nas três folgas, inclusive na mais apertada; o sazonal não certifica em
nenhuma. Isso confirma e **precisa** a fronteira nomeada no lote 1.

## O que muda o quadro: **duas refutações CERTIFICADAS**

No lote 1 a única refutação veio com `TIME_LIMIT` — conclusiva, mas sem prova do
lado do SCIP. **No lote 2 há duas com status OPTIMAL:**

| | caso 44 | caso 45 |
|---|---|---|
| célula | cancelamento regular, folga 5e−2 | cancelamento regular, folga 1e−3 |
| **modo** | `:zeroed` | **`:innovations`** |
| Ipopt | 57,868044 | 54,706978 |
| **SCIP** | **57,536983** | **53,681295** |
| **Ipopt está acima do ótimo em** | **0,575%** | **1,911%** |
| status / gap | OPTIMAL, 1,09e−09 | OPTIMAL, 1,07e−08 |

**São certificados de otimalidade global.** Não é "o SCIP achou algo melhor": é
"o ótimo global é este, provadamente, e o Ipopt não chegou nele".

**E o caso 45 está no modo `:innovations` — o que o paper defende.** Não dá para
tratar a refutação como artefato do `:zeroed`.

### O que **não** se pode dizer, e eu quase disse

O caso 45 **não é o caso 25 do lote 1 com outro modo.** O gerador semeia por
`MersenneTwister(1000 + id)`, então **id diferente ⇒ série diferente**. São dois
sorteios da mesma família de DGP, não a mesma série sob dois modos. A comparação
direta modo-a-modo **não foi feita** e não deve ser insinuada.

## Célula E — a fronteira vai até T = 480, e começa a ceder em 960

| T | resultado |
|---|---|
| 480 | **OPTIMAL** em 136 s |
| 960 | 1 **OPTIMAL** em 366 s · 2 TIME_LIMIT |

Nos três de `T = 960` o incumbente do SCIP **coincide com o ponto do Ipopt**. A
dimensão volta a pesar, mas só uma ordem de grandeza acima do que a M4 mensal usa.

## Limitação do desenho, encontrada: **nem todo TIME_LIMIT rende certificado**

Um caso de `T = 960` terminou com `dual bound = −1e+20` — a sentinela do SCIP para
*nenhum limite dual estabelecido*. O gap relativo calculado dá **1,07e+17**, que
não é um certificado frouxo: **é a ausência de certificado**.

**O produto "gap-no-timeout" não é universal**, e a tabela precisa distinguir três
estados, não dois: certificado, certificado parcial quantificado, e **sem limite
dual**. Um em quatro casos de `T = 960` caiu no terceiro.

**Correção:** o gap só é reportável quando `dual > −1e19`. Aplicado à leitura;
nenhum número anterior muda, porque em todo o lote 1 o dual era válido.

## Célula G — `:innovations` fora do caso adversarial

`inov-T120`: **OPTIMAL** em 20 s, gap 1,6e−09, Ipopt coincide a 1e−4.
O modo certifica normalmente onde não há quase-cancelamento.

---

# Como isto entra no paper — decidido 22/08

## Os três estados do certificado vivem na TABELA, não em rodapé

Reportar um "gap" de 1,07e+17 como certificado frouxo seria **falso**.
`dual bound = −1e+20` é a sentinela do SCIP para *certificado inexistente*.
A coluna passa a ter três estados, com contagens honestas:

| estado | condição | o que se pode afirmar |
|---|---|---|
| **certificado** | `OPTIMAL` | este é o ótimo global |
| **certificado parcial** | `TIME_LIMIT`, dual finito | nada mais que X% melhor existe na caixa |
| **sem certificado** | `TIME_LIMIT`, sem dual | **nada** |

**1 de 4 casos em `T = 960` caiu no terceiro estado.** O rodapé explica a
sentinela; **o estado vive na tabela.**

Isto é um **limite medido do próprio dispositivo**: o gap-no-timeout converte
células condenadas em produto **apenas quando o SCIP estabelece limite dual**, e
em `T = 960` às vezes não estabelece.

## A seção de otimalidade fica mais forte por ficar menos confortável

Abre com o **instrumento funcionando**, não com dano contido: a célula
adversarial existia para pôr a certificação exatamente onde ótimos locais eram
plausíveis — **e os achou, com prova.**

> Em N casos o ótimo do Ipopt é certificadamente global; em **2 casos
> adversariais de quase-cancelamento o Ipopt está provadamente acima do global**
> (0,575% e 1,911%, certificados a 1e−09 e 1e−08), **inclusive no modo
> `:innovations`**.

**Sem hedge no modo.** Não dá para tratar como artefato do `:zeroed`.

### A frase de escopo, honesta nos dois sentidos

Os gaps certificados são sobre o **objetivo CSS**. O efeito, em previsão, de estar
1,9% acima do global numa região fracamente identificada **não foi medido**.

**Escreve-se "não medido" — nunca "provavelmente irrelevante para previsão".**

## O fio transversal, nomeado uma vez

Esta é a **terceira aparição independente** da região de quase-cancelamento neste
trabalho:

1. o **mínimo do log-determinante** (Seção 4);
2. os **atratores de fronteira** do braço D;
3. agora, **armadilhas de ótimo local certificadas**.

> *O quase-cancelamento é o terreno patológico recorrente da paisagem CSS.*

Uma frase amarra as duas seções mais duras do paper — e **justifica o multistart
do pacote como necessidade medida, não ornamento.**

## Disciplina de leitura

A **pontuação formal da predição do lote 2 acontece uma vez, em 14/14, na leitura
de 24/08 junto com o lote 1.** O que foi lido a 12/14 é status mais a resposta da
célula F à pergunta que o lote 1 abriu — leitura legítima porque a célula estava
**completa** e o desenho **carimbado**.
