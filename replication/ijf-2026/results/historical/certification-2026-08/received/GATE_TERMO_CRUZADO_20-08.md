# Gate — termo cruzado do determinante no critério exato

**Escrito em 20/08, ANTES de qualquer implementação e de qualquer medição.**
Nenhum número deste experimento foi visto. Caixa de tempo do revisor
metodológico: gate verde até **23/08**, decisão de disparo da M4 em **24/08** pela
projeção de tempo de parede, M4 até **27/08**, escrita **28–31/08**. Perdida
qualquer data, o paper embarca com o resultado matemático e **sem alegação de
previsão**.

---

## 0. O que se implementa

O objetivo do critério exato já é `S × fator`, com

    fator = ∏_j (1−κ_j²)^(−j/T)          [bloco AR]
          × ∏_j (1−κMA_j²)^(−j/T)        [bloco MA]
          × ∏_j (1−κSMA_j²)^(−s·j/T)     [bloco MA sazonal]

isto é, `|Ω|^{1/T}` montado por blocos. **O termo cruzado entra na mesma forma**,
como mais um produto de potências, e — ao contrário dos blocos — direto nos
coeficientes, sem coeficientes de reflexão e sem raízes. Para
`(1,·,1)(1,·,1)[s]`:

    cruzado = (1+ϕθ)^(2/T) · (1+ϕˢΘ)^(2/T) · (1−Φθˢ)^(2/T) · (1+ΦΘ)^(2s/T)

Verificado contra o `logdet` numérico a 1e−13 (ver `ACHADO_DETERMINANTE.md`).

## 1. G0 — **A COMPARABILIDADE, e ela pode bloquear a M4**

A forma acima é para `p,q,P,Q ≤ 1`. Para ordens maiores o termo existe e é
polinomial — `∏_i(1−a_iˢ b)` é simétrico nas raízes inversas do AR regular, logo
polinômio nos coeficientes via identidades de Newton — mas a expansão em `s = 12`
é trabalho real e não está feita.

**Aplicar o termo só a algumas ordens recria exatamente o defeito da Seção 5.**
Candidatos com o termo e candidatos sem ele seriam pontuados sob critérios
diferentes, e a seleção deixaria de comparar como com como. Foi assim que
corrigir *metade* dos blocos piorou a seleção mais do que não corrigir nenhum.

**Portanto, G0 é pré-condição de tudo:** ou o termo cobre **todas** as ordens que
a busca visita, ou a rodada da M4 **não usa busca** — ordem fixada, e a alegação
passa a ser sobre estimação, não sobre seleção.

Decisão pré-registrada, tomada agora: se a expansão de Newton para ordens
superiores não estiver pronta e verificada até **22/08**, a M4 roda com **ordem
fixada** (a ordem selecionada pela configuração pinada de paridade), e o
resultado é reportado como **efeito de estimação com ordem fixa**, nunca como
efeito de seleção. Nenhuma alegação de seleção sem cobertura total.

## 2. G1 — corretude em JuMP

O termo escrito como expressão do solver, avaliado num ponto, bate com
`logdet` numérico da covariância a **≤ 1e−6**, numa grade de pelo menos 12
combinações de `(ϕ,θ,Φ,Θ)` incluindo sinais mistos.

*Mata:* erro de transcrição da álgebra para o código. Precedente nesta sessão: um
gate ficou keyed numa variável que só existia sob `invertible=true`, e a
medição correspondente foi inválida sem que nada na saída denunciasse.

## 3. G2 — o limiar `T*`, **pré-declarado aqui**

A forma é a constante **assintótica**. O erro decai geometricamente em `T`, com
taxa governada pelo maior módulo de raiz inversa do lado MA.

**Critério, fixado antes de medir:** o termo só é aplicado quando

    T ≥ T*  com  T* = 120

e, abaixo disso, o ajuste usa o critério sem o termo cruzado, com o fato
registrado em `model.metadata`. A justificativa do valor, e ela é anterior ao
resultado: na varredura já feita, `T = 120` mantém o erro em ≤ 2e−10 nos casos de
raiz moderada e ≤ 9e−6 no caso de `Φ = 0,95`; o caso ruim (`Φ=0,8, Θ=−0,9`) ainda
erra 0,19 em `T=120`, e **isso fica declarado como limitação, não corrigido por
escolha de limiar**. Mover `T*` depois de ver o OWA é botão, e está proibido.

Consequência operacional, aceita: na M4 mensal há séries com `T < 120`, e nelas o
braço novo coincide com o atual. A fração exata é medida e reportada.

## 4. G3 — a caixa, e a barreira do lado cruzado

**Achado hoje, e é condição de implementação.** O termo cruzado diverge quando
`α_iβ_j → 1`, que no bloco sazonal é `1+ΦΘ → 0`. O pacote tem
`DEFAULT_DOMAIN_MARGIN = 1e-6`, que dá `1+ΦΘ ≥ 2e−6` e portanto
`2s·log(1+ΦΘ) ≈ −315` — finito, mas grande.

**E a margem só vincula com a parametrização por construção ligada.** As rodadas
de paridade usam `invertible = false`, que é o default e o comportamento do R, e
aí o lado MA é livre: o termo cruzado fica **ilimitado por baixo**.

Critério: ou o termo exige `invertible = true`, ou se verifica que ele é a
própria barreira do lado cruzado — como o determinante do bloco MA é a barreira
de invertibilidade. **A segunda hipótese não está verificada e não se assume.**
G3 passa quando um dos dois estiver estabelecido por medição, com 0 divergências
do Ipopt numa varredura que inclua o canto.

## 5. G4 — mecanismo, com verdade conhecida

Em simulado `(1,0,1)(1,0,1)[12]` com coeficientes conhecidos, o braço com o termo
cruzado **fecha a distância de estimação** contra a ML do R.

**Direção predita: fecha.** Referência: hoje a distância mediana CSS→ML é 21,6
nessa classe, contra ~0 nos blocos puros.

*Mata:* implementar o termo com sinal trocado — que produziria números plausíveis
e conclusão invertida.

## 6. G5 — segurança da rodada

- **0 falhas duras** no braço novo que não existam no braço base.
- O braço com o termo roda **ao lado** do CSS default, **nunca no lugar**. A
  configuração pinada de paridade (OWA 0,9039) não é tocada e não fica refém
  deste experimento.
- Kwarg novo com default preservando o comportamento atual **bit-a-bit**,
  verificado nos 16 dígitos.
- Carimbo de proveniência (`git rev-parse --short HEAD` + estado da árvore) em
  toda saída de medição.

## 7. G6 — projeção de tempo de parede

Medida no gate, sobre pelo menos 20 ajustes representativos: custo por ajuste ×
48.000, com o paralelismo real da máquina. **Critério de disparo: a projeção tem
de terminar até 27/08.** Não terminando, não dispara — e isso não é falha do
método, é a caixa de tempo funcionando.

---

## As três frases de desfecho, escritas antes do disparo

- **Melhora** → candidato a resultado principal do paper.
- **Piora** → confirma a leitura da Seção 5 com o elo fechado por ML **exata** em
  vez de aproximada, que é a versão mais forte dessa seção.
- **Empata** → a forma fechada é resultado matemático de efeito neutro.

**A predição honesta, pelos quatro casos limpos já medidos, é que piore.**
Escrevê-la antes é o que faz deste um gate. O sucesso do experimento é **gate
verde mais M4 completa**; a direção do OWA é resultado, não critério.

## Regra de parada desta frente

**Um gate, um corte, zero iteração.** Gate falhou ou prazo estourou: o resultado
matemático entra no paper, a implementação não, e ninguém mexe no gate. Esta
frente não conta contra a regra de parada da grade exógena, que está completa e
lida.

---

# RESULTADO DO GATE — 20/08, mesma noite

| gate | resultado |
|---|---|
| **G0** cobertura de ordens | decidido: escopo `≤1`, guard mudo, M4 com ordem fixada se necessário |
| **G1** corretude em JuMP | **PASSA** — 12/12 casos, erro máximo **1,3e−11** contra critério de 1e−6 |
| **G2** limiar `T*` | pré-declarado, `T* = 120`, constante documentada no código |
| **G3** limitação por baixo | **PASSA** — ínfimo de `log\|Ω\|` é **0,0000**, estável de margem 1e−2 a 1e−8 |
| **G4** mecanismo | **FALHA** |
| G5, G6 | não executados (G4 corta) |

## G1 — o termo está certo

O teste compara o **código**, não a fórmula: com coeficientes fixos, `S` é idêntico
com e sem o termo, logo `T·log(obj_com/obj_sem)` é exatamente o cruzado. Contra o
cruzado extraído do `logdet` numérico (raízes explícitas da estrutura
multiplicativa, sem root-finder):

12 casos com sinais mistos, erro de **1,5e−14 a 1,3e−11**.

## G3 — limitado por baixo, e por razão estrutural

Busca aleatória densa (400.000 pontos por margem) do ínfimo de `log|Ω|` na caixa
admissível. O ínfimo é **exatamente 0,0000** e não se move ao encolher a margem de
`1e−2` para `1e−8`. O argmin converge para um ponto **interior**,
`(0,3165; −0,3183; 0,5466; −0,5458)` — onde `θ ≈ −ϕ` e `Θ ≈ −Φ`, isto é,
cancelamento quase exato, processo quase ruído branco.

Ou seja `log|Ω| ≥ 0` sempre, com mínimo no ruído branco. **O termo não é
sumidouro.**

## G4 — FALHA, e o modo de falha é o resultado

Simulado `(1,0,1)(1,0,1)[12]`, `T = 200`, verdade `(0,6; 0,4; 0,7; −0,4)`, 10
réplicas, alvo externo `stats::arima(method="ML")` do R nos mesmos dados.

| | distância mediana ao R |
|---|---|
| sem o termo | **0,2076** |
| com o termo | **0,4294** |

**Fecha em 3 de 10 réplicas.** O termo **afasta** da ML, não aproxima.

E o modo de falha é diagnóstico: **em 6 das 10 réplicas `Φ` e `Θ` vão para a
fronteira, `1,0` e `−1,0`** — a crista de cancelamento, onde os polinômios
sazonais se anulam. Idêntico com `invertible = true` e com `invertible = false`.

### A leitura, e a parte medida separada da conjecturada

**Medido:** `log|Ω|` tem mínimo no cancelamento (G3), e o termo somado ao objetivo
torna essa crista **atrativa**. O G3 estabeleceu corretamente que o termo é
limitado por baixo; a conclusão operacional que tirei dali — que portanto não
haveria problema de fronteira — estava errada. O risco nunca foi divergir; era o
mínimo estar na crista.

**Conjecturado, não medido:** a verossimilhança verdadeira também tem a crista de
redundância ARMA, mas ali ela é **plana**, não picada — o modelo é não
identificado, não preferido. Se o nosso termo a torna picada, a explicação
natural é que a constante **assintótica** é má aproximação exatamente onde o
solver é atraído: no cancelamento `|Φ|, |Θ| → 1`, os módulos das raízes inversas
vão a 1, e é precisamente aí que a varredura em `T` mostrou o erro maior (0,55 em
`T=60` já com `|Θ| = 0,9`). **Isto é hipótese e está declarado como tal.**

## Decisão, pela regra pré-registrada

**Um gate, um corte, zero iteração.** G4 falhou:

- a **implementação não embarca**;
- o **resultado matemático embarca**, como já estava previsto incondicionalmente:
  a forma fechada, a verificação em sete ordens, o qualificador assintótico, e a
  linha de que perfilamento/burn-in não alcança o determinante;
- a Seção 4 ganha uma **quinta linha no mapa**, e ela é conteúdo novo: *o termo é
  expressável e correto, e ainda assim não é utilizável como está, porque torna
  atrativa a crista de cancelamento que a verossimilhança verdadeira deixa plana*;
- **ninguém mexe no gate.**

A branch `davi/cross-determinant-term` fica local, não vira PR, e o
`objectiveValue` em metadata — que é diagnóstico legítimo e independente — vai
separado.

---

## Completação do G4 — o arco inteiro, registrado

O relatório do gate afirmava *"idêntico com `invertible = true`"*. **A afirmação
era tecnicamente verdadeira e substantivamente vazia:** o flag não vinculava, logo
a condição que a análise prévia nomeara nunca foi exercida.

Achado pelo **controle de vinculação do próprio gate**, rodado porque resultados
idênticos sob duas configurações é a assinatura de flag inerte — e essa assinatura
já havia pego dois outros casos no mesmo dia.

| `exactCrossTerm` | `invertible` | `metadata[crossTerm]` | objetivo |
|---|---|---|---|
| false | false / true | ausente | 55,423832 |
| true | false / true | **aplicado** | 27,582524 |

O termo **vincula**; o `invertible`, não.

**Completado, e o resultado é vazio.** Em precisão total, `Φ = 0,99999901` e
`Θ = −0,99999901` — a margem documentada `1−10⁻⁶`, não `1,0` literal. A caixa
vincula nos dois lados, com e sem o flag. Para `q = Q = 1` a caixa **coincide**
com a região de invertibilidade, logo o flag nada tem a remover; para `q ≥ 2` ela
não coincide (a região do MA(2) é o triângulo `θ₂ ± θ₁ < 1`), e ali não foi
testado.

**O arco, para o registro:** reportado sem teste → pego pelo controle de
vinculação → completado → vacuoso → **o corte se sustenta pelo mérito**, e agora
sem asterisco.

E uma reclassificação: o `invertible` inerte, que eu havia levantado como possível
décimo defeito, é **redundância de primeira ordem**, não violação de contrato. Vai
ao mantenedor como abrangência pendente, fora do paper.

**A §4.5 fica congelada** após os dois ajustes desta completação. Achados
posteriores desta frente vão para o registro e para o entregável do mantenedor,
não para o paper — o rascunho precisa parar de se mover para a passada de
consistência valer.

---

# G4′ — gate NOVO sobre composição diferente. Pré-declarado em 21/08, antes de rodar.

**Não é iteração do G4.** O G4 cortou corretamente a composição que mediu:
determinante + forma quadrática **CSS-perfilada**. Esse corte fica de pé. O G4′
mede composição **diferente**: determinante + forma quadrática **exata via
inovações**, justificado por evidência de caminho decisiva (o critério exato tem
mínimo na verdade e sobe rumo ao canto — não é o critério que é patológico).

## Os quatro braços

| braço | forma quadrática | termo cruzado | distância mediana ao R |
|---|---|---|---|
| **A** | `:penalized` (antiga) | não | **0,2076** (medido) |
| **B** | `:penalized` (antiga) | sim | **0,4294** (medido) |
| **C** | `:innovations` (nova) | não | a medir |
| **D** | `:innovations` (nova) | sim | a medir |

## Critérios, fixados antes

- **Primário — D fecha:** distância mediana ao ponto de ML do `stats::arima`
  **< 0,2076**, com os **quatro coeficientes estritamente interiores** às caixas
  (print em precisão total; se algum sentar no limite, a rodada **não conta**).
- **Secundários:** `C − A` isola o efeito da parametrização; `D − C` isola o
  efeito do determinante **sob pareamento correto**.
- Mesmas 10 sementes, mesmo alvo externo, **zero botões novos**.

## Predições, escritas antes de rodar

**D fecha; C sozinho move pouco; o determinante só ajuda dentro de D.**

## Escopo obrigatório da alegação, qualquer que seja o desfecho

O controle de deriva (21/08, 10 pontos interiores) deu razão a `y'Ω⁻¹y` dentro de
**1,6%**, e a deriva rumo ao canto vai a **1,71**. O `:innovations`
**super**estima perto do canto, o que torna o canto **menos** atrativo do que a
ML exata tornaria — **a deriva é protetora**. Logo, se D fechar, parte do
fechamento pode vir da sobre-repulsão e não do pareamento correto.

**A alegação nunca é "ML exata".** É: *casa com a ML no interior (1,6% em dez
pontos); desvia conservadoramente perto do canto.*

## Frases de desfecho, pré-escritas

- **Fecha:** *"o termo exige o par dele, e eis o par — medido"*, com o escopo acima.
- **Falha:** *"não utilizável na formulação do pacote; a causa de pareamento é
  consistente com o caminho medido, e nem o par natural resgata"* — o corte fica,
  e o diagnóstico fica **certo** em vez de errado.

As duas são publicáveis. É por isso que o teste não é aposta.

---

# RESULTADO DO G4′ — a tabela 2×2 completa

| braço | forma quadrática | termo cruzado | distância mediana à ML do R |
|---|---|---|---|
| **A** | `:penalized` | não | 0,2076 |
| **B** | `:penalized` | sim | 0,4294 |
| **C** | **`:innovations`** | não | **0,0863** |
| **D** | `:innovations` | sim | 0,7984 |

- **C − A = −0,1213.** A parametrização por inovações **sozinha fecha 58% da
  distância** à máxima verossimilhança, sem termo de determinante nenhum.
- **D − C = +0,7121.** O determinante piora, mesmo sob o pareamento correto.
  **9 de 10 réplicas** com coeficientes na fronteira.

**Primário: FALHA.** `D = 0,7984` não fica abaixo de `A = 0,2076`, e só 1 de 10
tem os quatro coeficientes estritamente interiores.

## A predição pré-registrada errou nas duas metades

Escrito antes de rodar: *"D fecha; C sozinho move pouco; o determinante só ajuda
dentro de D."* **C moveu muito e D não fechou.** Fica no registro como falseada
nos dois sentidos.

## O que o 2×2 encerra

**O termo cruzado falha sob os dois pareamentos** — com a forma antiga (+0,2218)
e com a exata (+0,7121). O corte da §4.5 fica, e agora se apoia na tabela
completa em vez de inferência: não era problema de composição com a forma
quadrática.

## O que o 2×2 abre, e é maior

A **parametrização por inovações** é ganho grande, limpo e isolado: 58% do gap
CSS→ML fechado **removendo** variáveis de decisão, não acrescentando termo. É um
quarto degrau na escada da Seção 3, e o único que casa com a máxima
verossimilhança.

A frente perseguiu a direção certa e o alvo errado: era a **forma quadrática**,
não o determinante.

## Fechamento do registro do G4′ — interioridade e fronteira

**(a) O braço C está estritamente interior**, em todas as réplicas. Exemplos em
precisão total: `(0,646175; 0,471054; 0,566908; −0,194474)`,
`(0,589203; 0,492855; 0,622257; −0,199642)`. Nenhum coeficiente perto de
`1−10⁻⁶`. A alegação de estimação do quarto degrau está limpa.

**(b) O braço D pina sempre no par SAZONAL**, `Φ = 0,999999` e
`Θ = −0,999999`, em 9 de 10; na décima, só `Φ`. **Nunca no par regular.** É o
mesmo canto de cancelamento sazonal do G4 original — **o atrator sobrevive à
sobre-repulsão** que o `:innovations` introduz perto do canto.

E um sinal que não estava no G4: a maioria dos ajustes do D termina em
`SLOW_PROGRESS`. O solver não converge no canto — **para** lá.

### A quarta instância de medir-num-lugar-e-generalizar

O teste de caminho que reabriu esta frente percorreu verdade→canto **com `ϕ` e
`θ` fixos na verdade**, variando só o par sazonal. Nessa fatia o critério composto
é bem-comportado: mínimo na verdade, subida monótona. **E ainda assim o otimizador
encontra o atrator em 9 de 10, movendo os quatro coeficientes.**

**Uma fatia unidimensional não certifica uma paisagem de quatro dimensões.** É a
quarta instância desta rodada do mesmo padrão, e é a razão de esta seção passar a
enunciar tabelas e escopos em vez de interpretações.

---

# GATE DAS 1.500 — cadeia de precondição fechada, telemetria pré-declarada

**21/08.** O bloqueador nunca foi custo — foi consistência de nível. A cadeia:

| elo | resultado |
|---|---|
| `c` e `trend` na restrição do `zpre` | **implementado** (o `ŷ` em-amostra os inclui; a restrição os omitia) |
| teste da média 50, `allowMean=true` | **passa** — 4/5 acompanham o R e o `:free`; 1 discrepante |
| controle de deriva, configuração que vai rodar | **passa** — 10/10 dentro de 2%, desvio máximo 1,6% |
| consistência em série real, 5 da M4, dois modos | **passa** — sem patologia; `p=P=0` dá idêntico ao dígito |

**Configuração congelada:** `presampleBurnIn = 12` — a validada. Eu havia mudado
o default para 0 **depois** de validar, com base em uma série; revertido.

## O custo: as quatro atribuições que morreram

O "312×" era **compilação JIT do caminho novo, paga uma vez por processo**.
Medido rodando 5 séries no mesmo processo: 13,3s na primeira, depois **0,085 /
0,234 / 0,347 / 0,050**. Na mesma série, `:penalized` custa 0,183s e
`:innovations` **0,050s** — o modo novo é mais rápido.

Explicações que ofereci e que a medição derrubou, em ordem: hessiana densa
(`ε` sempre foi variável de decisão); janela inflada pelo burn-in (burn-in 0 mede
igual); custo de montagem 170× (era JIT, some com aquecimento por modo). **As
quatro tinham a mesma causa: n = 1 generalizado.**

## Telemetria pré-declarada, antes de rodar

Logging **observacional**, não toca critério de gate nenhum. Por ajuste, nos dois
braços: **iterações do solver, status de convergência, tempo de parede.**

Motivo: `19 iterações contra 200` é resultado real e independente de acurácia,
mas hoje tem **n = 1**. Com a telemetria, entra com n = 1.500. Até lá o escopo é
*"esgotou o teto de iterações nos candidatos de uma série"*, **nunca** "não
converge".

Dois guardas escritos agora: **teto atingido ≠ previsão ruim** — superfícies CSS
são planas perto do ótimo, então status de convergência tem de ser pareado com
sMAPE antes de qualquer alegação; e **a paridade de 0,9065 fica observação**, sem
reinterpretação antes de o dado existir.

## Predição, inalterada

**"Piora ou nulo"**, pelos quatro casos da Seção 5. As cinco séries do controle
de consistência deram todas ≤ 0, mas **n = 5 não é evidência** e não altera a
predição registrada.

## Confirmação da cadeia e do binário — exigência do revisor, registrada

**Cadeia de consistência de nível: FECHADA** antes do disparo.

| elo | resultado |
|---|---|
| sMAPE do `:penalized` no sid 38710 | 38,850817 — **idêntico** ao `:innovations`; a ordem selecionada tem `p=P=0`, onde os modos coincidem por construção, então o elo passa mas **não discrimina** |
| teste da média 50, `allowMean=true` | **4/5** acompanham o R e o `:free` (c ≈ 25,0/20,0/16,5/25,6 contra 25,3/20,3/16,8/25,9 do R); 1 réplica discrepante (c=12,9 contra 18,2) |
| `c` e `trend` na restrição do `zpre` | implementado |
| controle de deriva na configuração congelada | **10/10** dentro de 2%, desvio máximo 1,6% |
| 5 séries reais da M4, dois modos | sem patologia de nível; `p=P=0` dá idêntico ao dígito |

**Binário congelado: `f90e554`.** Verificado que `src/` não foi tocado entre o início
das rodadas (07:24) e a escrita das saídas (07:47 e 07:58) — gate e censo executam
a mesma implementação.

**Guarda registrado:** a regra de "a M4 morre se o gate reprovar" corre **só para
frente**. A leitura do gate é **cega ao custo afundado** das 48.000 —
computação afundada não é evidência.
