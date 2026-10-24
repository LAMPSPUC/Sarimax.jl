# Lote de certificação global com SCIP — pré-declarado em 21/08, antes de rodar

**O pacote não é tocado.** A reformulação bilinear vive **só no arnês externo**.
No pacote fica como backlog, com motivação dupla (aceitação do SCIP e possível
custo de diferenciação automática).

## O que os testes rápidos já estabeleceram

| caso | SCIP | Ipopt |
|---|---|---|
| ARMA(1,1), T=40 e 60, 6 réplicas | **OPTIMAL** em segundos | **6/6**, gap ~1,5e−6 |
| `(1,0,1)(1,0,0)[12]`, T=60 | TIME_LIMIT em 120s | incumbente igual a 4,9e−14 |
| `(1,0,1)(0,0,0)[12]` `:free` | TIME_LIMIT | igual a 3,8e−11 |
| qualquer `Q > 0` | **recusa** | — |

A recusa é nomeável: `ScalarNonlinearFunction-in-EqualTo not supported`. O SCIP via
MOI aceita **quadrático**; o termo `θ_j·Θ_w·ε_{t−j−sw}` é **trilinear**.

## O prior honesto, escrito antes

**O gargalo é a dureza do branch-and-bound, não a aceitação do modelo.** O caso
sazonal **sem** `Q>0` — que o SCIP já aceita — **já bate no limite**. A
reformulação bilinear conserta a *aceitação* do `Q>0`; ela **não** faz nada pelo
gap que não fecha.

**Predição registrada: a certificação sazonal provavelmente não fecha, mesmo
reformulada.** O rendimento realista são duas coisas, ambas publicáveis:

- **(a)** casos de ordem baixa certificados em escala;
- **(b)** um **negativo medido** sobre certificação sazonal dentro de orçamento
  declarado.

**Nenhum dos dois é "certificamos o SARIMA".**

## Desenho, congelado

- **Ordens:** `(1,0,1)(P,0,Q)[12]` com `P,Q ∈ {0,1}` — quatro combinações.
- **`T ∈ {60, 120}`**, `:zeroed`, 5 réplicas por célula. **40 casos.**
- **Limite de tempo: 600 s por caso.** Declarado agora e **nunca estendido depois
  de ver resultado**.
- **Reformulação bilinear** no arnês externo: `u = θΘ` como variável com restrição
  quadrática, depois `u·ε` quadrática.
- **Controle de consistência externo-vs-pacote:** o objetivo do arnês tem de bater
  com o do pacote nos mesmos coeficientes. Já demonstrado a **4,9e−14** nos casos
  que o SCIP aceita sem reformulação.

## O que se reporta

**M de N certificados, com a fração em TIME_LIMIT**, por célula. A frase-alvo:

> em N séries de ordem ≤ X e T ≤ Y, o ótimo do Ipopt é certificadamente global em
> M de N

**Nunca** *"o Ipopt acha o global"*.

## Guardas de redação

- *"incumbente idêntico"* **não é certificado**. Onde o SCIP bate no limite, o que
  se diz é que ele achou o mesmo ponto sem provar.
- A frase de contraste com espaço de estados suaviza de *"não tem o que entregar a
  um solver global"* para **"entrega apenas avaliações de função"** — a própria §4
  deste paper mostra que a recursão é expressável como restrições em princípio, e
  a versão categórica seria refutada pelo nosso próprio mapa.
- **30 minutos de literatura antes de qualquer sabor de novidade.** Otimização
  global e MINLP para estimação ARMA existem em alguma forma; a alegação escopa a
  *"para esta formulação"*.

## Calendário

**Leitura em 25/08.** Não fechando até lá, embarca o fallback: a frase da discussão
com os números dos testes rápidos e zero código novo. **Esta é a última adição.**

---

# Defeito de instrumento, P1 anulada, P2 registrada — 21/08, antes da leitura

## O defeito

O arnês declarava `@variable(m, ε[1:T])` — **sem bounds**. São `T` variáveis livres
(60 ou 120) contra quatro coeficientes. Branch-and-bound constrói relaxações a
partir de bounds; com os `ε` livres, a relaxação de `Σε²` é fraca e as `T`
variáveis dominam a árvore.

É defeito de instrumento da **mesma categoria** da caixa `[-2,2]` que admitia a
região não-invertível — e foi descoberto **antes** de o lote completar.

## A régua adotada, que é também a regra de parada

**Só entram bounds que são consequência provável de (dados, incumbente, região
admissível). Qualquer bound que exija constante escolhida fica fora.**

- **`ε` limitado por teorema:** qualquer solução ao menos tão boa quanto o
  incumbente satisfaz `Σε² ≤ f_Ipopt`, logo **`|ε_t| ≤ √f_Ipopt`**. Derivado, não
  escolhido. É *optimality-based bound tightening*, técnica padrão. O
  `k·std(y)` que eu havia cogitado **morre pela mesma régua**: `k` é escolhido.
  Os `ε` pré-amostrais já estão **fixados em zero** no arnês, então o vetor fica
  inteiramente limitado — meio-limitar e chamar de limitado seria a próxima
  correção da série.
- **Incumbente do Ipopt como warm start:** transferência de informação, não botão.
  A obrigação de prova continua sendo o **dual bound do SCIP**.
- **Produtos por envoltória derivada das caixas dos fatores**, não `[-1,1]` fixo.

## O arco, com carimbo, porque a distinção é o método

> **P1 registrada** (*"a certificação sazonal provavelmente não fecha, mesmo
> reformulada"*) → **defeito de instrumento achado** (ε livres; caixa `[-2,2]`
> corrigida antes) → **P1 anulada, não pontuada** → **P2 registrada antes do lote
> apertado.**

Marcar P1 como "falseada" seria teatro; substituí-la em silêncio seria o pecado
que este projeto existe para impedir.

**E cai junto uma frase do revisor metodológico**, que estava premissada no mesmo
instrumento: *"o gargalo é dureza de B&B, não aceitação do modelo"*. Se o lote
apertado certificar os sazonais, ela cai com a P1, e o registro mostra que caiu.

## P2 — registrada agora, antes de qualquer leitura

**Genuinamente incerta, e escrita como incerta.** Com os `ε` limitados por `√UB` e
o incumbente entregue, não sei dizer se a certificação sazonal fecha em 600 s. O
bound do `ε` é uma redução grande do espaço de busca; a dimensão do problema
(`T + 4 + produtos`) continua alta. **Não aposto em nenhum dos lados**, e essa é a
razão de o lote valer a pena.

## Alegação, com o escopo que o incidente dos 293% obrigou

**O certificado é relativo à caixa declarada.** "Global" significa *global sobre a
região admissível — estacionária e invertível* — que é o domínio real do
estimador, logo é a alegação certa, mas o paper **diz isso explicitamente**.

E entra como asserção do arnês a checagem de sanidade que salvou o dia: **objetivo
certificado ≥ piso plausível de ruído**. Foi 17,4 estar abaixo do piso numa série
de 60 pontos que me fez olhar em vez de reportar "o Ipopt erra o global por 293%".
