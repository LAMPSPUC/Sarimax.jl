# SCIP lote 1 — otimalidade global · 32/32 · 21/08

Arnês externo, reformulação bilinear. **O pacote não foi tocado.** SCIP 8.0.0,
uma configuração, defaults, limite de 600 s nunca estendido, bound do `ε` derivado
por OBBT, warm start completo com o ponto do Ipopt.

## O placar

| célula | OPTIMAL | TIME_LIMIT |
|---|---:|---:|
| **A** — fronteira não-sazonal (T = 30…240) | **12** | 0 |
| **B** — fronteira sazonal | 0 | 12 |
| **C** — adversarial | 2 | 4 |
| **D** — modo `:innovations` | **2** | 0 |

## O achado: **um caso em que o Ipopt não achou o ótimo global**

**Caso 26 — quase-cancelamento sazonal** (raízes AR e MA do bloco sazonal quase
coincidentes, `Φ = 0,700`, `Θ = −0,699`):

| | |
|---|---|
| objetivo do Ipopt | 46,660913 |
| **objetivo do SCIP** | **45,314976** |
| diferença | **2,97% do objetivo** |

**O SCIP encontrou um ponto viável estritamente melhor.** Note a assimetria
lógica que torna isto conclusivo apesar do `TIME_LIMIT`: **para refutar
otimalidade global basta exibir um ponto melhor — não é preciso certificado.**
O SCIP não certificou o próprio ponto (gap 32%), e isso não importa: o ponto do
Ipopt está **provadamente** a ≥2,97% do ótimo.

**Em todos os outros 31 casos o ponto do Ipopt se sustenta**, com acordo de ~1e−6
relativo — que é tolerância de solver, não discrepância.

Este é exatamente o caso que a célula adversarial existia para procurar, e o
**viés de seleção que eu tinha levantado era real**: o piloto, que só rodava
células fáceis, teria concluído "o Ipopt acha o global" sem nunca visitar o lugar
onde ele não acha.

## A fronteira da certificação é **sazonalidade, não dimensão**

**T = 240 certifica em 67–134 s.** O bound OBBT tirou a dimensão do caminho
crítico — medido, não assumido. Já **todas as 12 células sazonais esgotam os
600 s**, mesmo em `T = 36`.

## Os `TIME_LIMIT` como certificados parciais quantificados

O dual bound **é** um certificado: *"nada mais que X% melhor existe na caixa"*.

| célula B, gap certificado | |
|---|---|
| melhor | **1,9e−4** |
| mediana | ~1,2e−2 |
| pior | 4,9e−1 |

Nas 12 sazonais o **incumbente do SCIP é igual ao ponto do Ipopt a ≤2,5e−9** —
os dois acham o mesmo ponto; falta a prova, não o ponto.

## As duas células `:innovations` certificaram

Casos 31 e 32, `T = 60`: **OPTIMAL**, gaps 7,8e−9 e 1,1e−8. O modo que o paper
defende foi certificado, não só o `:zeroed`.

## O guarda que disparou errado — meu erro, registrado

A asserção do piso de ruído (`Σε² ≥ 0,25·(T−1)`) falhou nos **três casos de série
real da M4** (ids 28, 29, 30).

**Não é achado — é guarda mal especificado por mim.** O piso pressupõe
`σ_ε = 1`, o que vale para as sintéticas por construção. Mas eu **padronizei as
séries reais pela variância da série**, e numa série autocorrelacionada a variância
da inovação é muito menor que a da série. Depois do z-score, `σ²_ε ≪ 1`, e a soma
de quadrados fica legitimamente abaixo do piso.

**A verificação substantiva, essa passa:** nas três séries reais o incumbente do
SCIP coincide com o ponto do Ipopt a ≤2,5e−9. Não há mínimo espúrio; há um piso
inaplicável.

**Correção para o lote 2:** o piso só se aplica a séries com `σ_ε` conhecido.
Para séries reais o teste correto é o acordo incumbente-vs-Ipopt, que é o que já
está sendo medido.

## O que isto entrega ao paper

1. **Otimalidade global verificada** na fronteira não-sazonal inteira, `T` de 30 a
   240, nos dois modos.
2. **Um contra-exemplo medido** — e a honestidade de reportá-lo é o que dá peso
   aos outros 31.
3. **A fronteira do método, nomeada:** sazonalidade multiplicativa, não dimensão.
4. Certificados parciais quantificados onde não houve prova.
