# Finalistas publicadas de Fernandes et al. (2025) nos dados F25

Script: [`run_f25_published_finalists.py`](../../run_f25_published_finalists.py)
(modelos em [`f25_published_finalist_models.py`](../../f25_published_finalist_models.py)).

## Para que serve

A reconstrução F25 (`paper1/`) ajustou CNNs novas por Optuna/TPE e não
recuperou as melhores execuções publicadas (melhor de 20: 66,6% PAA-CV e
73,8% PAA-FV, contra 86,7% e 82,1% publicados). Este job treina **as duas
arquiteturas que a fonte publicou** (Seção 4.4, Fig. 6) com exatamente o mesmo
F25: mesmos estados MATLAB, partição, realização do ruído, MinMax do treino,
PAA (583 × 10), 20 seeds de relatório e laço de treinamento. Assim o gap se
decompõe em

```
publicado − F25 ajustado (melhor de 20)
  = (publicado − finalista)            diferenças de dados/simulador/framework
  + (finalista − F25 ajustado)         resultado da otimização (HPO)
```

As colunas `gap_published_minus_finalist_best` e
`gap_finalist_best_minus_f25_tuned_best` do `summary.csv` são esses dois termos
(melhor de 20). A diferença de médias também está em cada `metrics.json`
(`comparison.gap_finalist_mean_minus_f25_tuned_mean`).

O braço com padrões do Keras (`PAA-PUB-KERAS__*`, seção abaixo) treina as
mesmas duas arquiteturas com a inicialização e o Adam do Keras. Ele separa, no
primeiro termo, o efeito desses padrões do framework.

**No artigo (31/10/2026).** Os resultados entraram no `paper1/` como controle:
parágrafo no desenho experimental, parágrafo e Tabela `tab:f25_finalists` nos
resultados, limitações, conclusão e uma frase no resumo. A tabela
(`paper1/tables/f25_published_finalists.tex`) é gerada a partir das matrizes de
confusão guardadas, sem treinar nada:

```bash
.venv-campaign-py313/Scripts/python.exe run_f25_published_finalists.py --paper-table
```

## Resultado (execução completa, 29/10/2026)

Foram 60 execuções na RTX 4070 Laptop, das 14:00 às 16:25, enquanto um TPE de
outro repositório rodava na CPU. A verificação de dados passou antes do treino
(`data_check.json`): os checkpoints F25 reproduziram 66,6% e 73,8%, com
matrizes de confusão idênticas.

Acurácia de teste em 10 classes nas 20 seeds, em %. DP amostral;
arredondamento *half-up* a partir das contagens das matrizes de confusão.

| Finalista | Média ± DP | Mín. | Mediana | Máx. | Execuções em 10% | F25 ajustado: média / máx. | Publicado |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PAA-PUB__S00` (CV) | 31,33 ± 19,08 | 10,0 | 26,35 | 59,3 | 6 | 57,33 / 66,6 | 86,7 |
| `PAA-PUB__S01` (FV, *same*) | 56,21 ± 11,57 | 10,0 | 57,95 | 65,5 | 1 | 46,69 / 73,8 | 82,1 |
| `PAA-PUB-VALID__S01` (FV, *valid*) | 59,49 ± 4,03 | 52,9 | 58,70 | 65,6 | 0 | 46,69 / 73,8 | 82,1 |

Decomposição do gap (melhor de 20, em pontos percentuais):

| Sensor | publicado − F25 ajustado | = publicado − finalista | + finalista − F25 ajustado |
|---|---:|---:|---:|
| CV | 20,1 | 27,4 | −7,3 |
| FV (*same*) | 8,3 | 16,6 | −8,3 |
| FV (*valid*) | 8,3 | 16,5 | −8,2 |

Leitura:

- **O termo de otimização é negativo nos dois sensores.** Com os mesmos dados,
  protocolo e seeds, as arquiteturas publicadas ficam abaixo das redes que o
  TPE do F25 encontrou. O gap não vem do resultado da HPO. Ele cabe inteiro, e
  com sobra, no termo de dados, simulador e framework. O braço Keras, abaixo,
  isola a parte que vem dos padrões do framework.
- **CV publicado.** 6 das 20 execuções colapsaram: a perda de validação ficou
  presa em ln 10 = 2,3026, com saída uniforme e todo o teste numa só classe.
  Outras 8 chegaram ao limite de 1.000 épocas ainda melhorando (melhor época
  ≥ 985), e esse limite segue a fonte. O máximo de 59,3% é, portanto, um limite
  inferior sob o protocolo publicado. As 14 execuções que aprenderam têm média
  de 40,46%.
- **FV publicado.** É mais estável que o FV ajustado (1 colapso contra 3;
  mediana 57,95% contra 53,50%), mas tem teto menor (65,5% contra 73,8%). O
  padding não muda o melhor run (65,5% com *same* e ceil, que reproduz o
  flatten 7.008; 65,6% com *valid*), e a variante *valid* não colapsou.
- **Trinca.** Como na reconstrução, o eixo de trinca é o gargalo: acurácia
  média de nível de trinca de 48,70% no CV e de 59,97% e 61,15% no FV.

O log do console desta execução (`full_run.log`) mostra a mediana do FV como
57,9 por arredondamento de float; o valor exato é 57,95%. O script agora
arredonda *half-up*. Os valores de referência são os de `metrics.json` e
`summary.csv`.

## Braço com padrões do Keras (30/10/2026)

A fonte treinou com TensorFlow/Keras. O F25 e as finalistas acima usam os
padrões do PyTorch. O braço `PAA-PUB-KERAS__*` treina as mesmas duas
arquiteturas publicadas (o FV com *same* e ceil) e troca só os dois padrões que
mudam a otimização:

- **Inicialização do Keras.** Pesos glorot_uniform e bias zero. No PyTorch, os
  bias são uniformes, e o limite dos pesos da camada densa do CV é 1/√27.840,
  2,4× menor que o limite glorot.
- **Atualização do Adam do Keras** (`KerasAdam`). O épsilon 1e-7 é somado a
  √v, e as correções de viés entram no tamanho do passo. O PyTorch soma o
  épsilon (1e-8) depois de corrigir v.

O laço de treino, os dados e as seeds são os mesmos. As 40 execuções levaram
41 min (30/10, 22:57–23:38), com o MATLAB de outro repositório rodando ao
mesmo tempo.

| Finalista | Média ± DP | Mín. | Mediana | Máx. | Execuções em 10% | No limite de 1.000 épocas |
|---|---:|---:|---:|---:|---:|---:|
| CV, PyTorch (`PAA-PUB__S00`) | 31,33 ± 19,08 | 10,0 | 26,35 | 59,3 | 6 | 8 |
| CV, Keras (`PAA-PUB-KERAS__S00`) | 43,59 ± 18,38 | 10,0 | 53,90 | 63,4 | 3 | 4 |
| FV, PyTorch (`PAA-PUB__S01`) | 56,21 ± 11,57 | 10,0 | 57,95 | 65,5 | 1 | 2 |
| FV, Keras (`PAA-PUB-KERAS__S01`) | 55,68 ± 9,50 | 20,3 | 55,25 | 67,8 | 0 | 0 |

Decomposição em três termos (melhor de 20, em pontos percentuais):

| Sensor | publicado − F25 ajustado | = publicado − finalista Keras | + finalista Keras − finalista PyTorch | + finalista PyTorch − F25 ajustado |
|---|---:|---:|---:|---:|
| CV | 20,1 | 23,3 | 4,1 | −7,3 |
| FV | 8,3 | 14,3 | 2,3 | −8,3 |

Leitura:

- **Os padrões do Keras estabilizam o CV publicado, mas fecham pouco do gap.**
  - A média do CV subiu 12,27 pontos (IC bootstrap de 95% da diferença: +0,6 a +23,3; Mann–Whitney p = 0,034).
  - A mediana passou de 26,35% para 53,90%.
  - Os colapsos caíram de 6 para 3, e as execuções que pararam no limite de épocas, de 8 para 4.
  - O melhor run, porém, só passou de 59,3% para 63,4%.
- **No FV não houve efeito consistente.** A diferença de média foi de −0,53
  ponto (IC −6,6 a +6,3; p = 0,34). O melhor run subiu 2,3 pontos, dentro da
  variação entre seeds.
- **Com os padrões de qualquer dos dois frameworks, as arquiteturas publicadas
  ficam abaixo das redes ajustadas do F25**: 63,4% contra 66,6% no CV e 67,8%
  contra 73,8% no FV. Também ficam 23,3 e 14,3 pontos abaixo dos valores
  publicados. Esse resto do gap fica nos dados e no simulador (modelo de
  trinca, malha do tabuleiro, partição). Inclui também as diferenças de
  framework e protocolo que não foram testadas aqui: a semântica do
  ReduceLROnPlateau e do EarlyStopping do Keras, o TF32 (ativo por padrão no
  TensorFlow em GPUs Ampere ou mais novas) e as escolhas que a fonte não
  relata.

## O que é treinado (5 × 20 execuções)

| ID | Sensor | Camadas | Feature map → flatten | Parâmetros | lr / batch |
|---|---|---|---|---:|---|
| `PAA-PUB__S00` | CV (S00) | Conv 48 k2, Conv 48 k3, *valid*, sem pooling; Dense 48 | 580 × 48 = **27.840** | 1.343.962 | 1e-3 / 24 |
| `PAA-PUB__S01` | FV (S01) | Conv 128/96/96 k3 *same*, MaxPool 2 *ceil* após cada; Dense 96 | 73 × 96 = **7.008** | 739.050 | 5e-4 / 32 |
| `PAA-PUB-VALID__S01` | FV (S01) | idem, *valid* + pooling *floor* (padrão do F25) | 71 × 96 = 6.816 | 720.618 | 5e-4 / 32 |
| `PAA-PUB-KERAS__S00` | CV (S00) | como `PAA-PUB__S00`, com inicialização e Adam do Keras | 580 × 48 = 27.840 | 1.343.962 | 1e-3 / 24 |
| `PAA-PUB-KERAS__S01` | FV (S01) | como `PAA-PUB__S01`, com inicialização e Adam do Keras | 73 × 96 = 7.008 | 739.050 | 5e-4 / 32 |

- Os números vêm de `core.f25_experiment_contract.PUBLISHED_PAA_CNN_TEMPLATES`;
  os flattens 27.840 e 7.008 conferem com o artigo (verificado a cada execução).
- O flatten publicado do FV (7.008 = 73 × 96) só sai com convoluções *same* e
  pooling com arredondamento para cima (583 → 292 → 146 → 73). A variante
  *valid* (71 × 96) é uma análise de sensibilidade.
- A camada densa de 96 unidades do FV fica fora da faixa de busca publicada
  (16–64); foi implementada como publicada.
- Protocolo idêntico ao F25: Adam, entropia cruzada, ReduceLROnPlateau (0,5;
  30 épocas; mín. 1e-6), early stopping 50, máx. 1.000 épocas, restaura os
  pesos de menor perda de validação, PyTorch determinístico.

## Por que os dados são os mesmos do F25

O script não modifica nada do F25 congelado: importa o contrato, o modelo e o
`training/f25_executor.py` e chama o próprio carregador (`_load_clean_selected`,
que confere o SHA-256 de cada `.mat`), o laço `_train_once`, `_evaluate` e
`_classification_metrics`. A cada execução ele também:

1. confere que os 5 arquivos-fonte que definem dados/modelo/treino são byte a
   byte iguais aos do bundle que gerou o F25 (commit `02b45a3`);
2. usa o `data_binding` registrado nos `run_record.json` dos jobs F25-R
   PAA__S00/S01 e o recibo `file_digests.json` do conjunto;
3. **re-pontua os checkpoints do melhor run F25** (PAA__S00 seed 1019 e
   PAA__S01 seed 1069) no teste preparado aqui. No smoke test deste laptop,
   ambos reproduziram 66,6% e 73,8% com **matrizes de confusão idênticas** e
   perda de teste com erro relativo ≤ 1,3e-6 (`smoke/data_check.json`). Se não
   reproduzir, o job aborta antes de treinar.

## Como rodar

Ambiente: o mesmo do F25 (`.venv-campaign-py313`: PyTorch 2.7.0+cu128,
numpy, scipy). A build cu128 também serve para a RTX 5060 Ti (Blackwell).

**Neste laptop (RTX 4070), recomendado.** O `f25.zip` na raiz do repositório
é detectado sozinho; os 10 `.mat` são extraídos para uma pasta temporária
local (não no OneDrive) e apagados no fim. No VS Code: abrir
`run_f25_published_finalists.py` e clicar Run. Ou no terminal:

```bash
.venv-campaign-py313/Scripts/python.exe run_f25_published_finalists.py
```

Se o `f25.zip` (ou um workspace F25 extraído) estiver em outro lugar, ajuste
`F25_SOURCE` no topo do script ou passe `--source <caminho>`.

**Em outro PC (RTX 5060 Ti, RTX 2060).** Extraia `f25/workspace` do `f25.zip`
(no PC da RTX 2060 basta usar `D:\f25\workspace`), copie os dois arquivos
`run_f25_published_finalists.py` e `f25_published_finalist_models.py` para a
raiz dessa pasta e rode da mesma forma com o ambiente de campanha. Ali o script
reconhece a pasta como workspace F25.

**Smoke test** (CPU, 2 seeds × 3 épocas, ~30 s, grava em `smoke/`):

```bash
.venv-campaign-py313/Scripts/python.exe run_f25_published_finalists.py --smoke
```

Opções úteis:
- `--no-valid-variant` pula a variante *valid*, e `--no-keras-arm` pula o braço Keras. Com as duas, rodam só as 40 execuções publicadas.
- `--finalists PAA-PUB__S00` roda só os finalistas indicados.
- `--device cpu` força a CPU.
- `--paper-table` só reescreve a tabela do artigo a partir dos resultados guardados.

Com 3 épocas, as execuções ficam perto de 10% (acaso); o smoke test só confere
o encadeamento.

**Interrupção** (suspensão, queda de energia): basta rodar de novo; seeds
terminadas ficam em `<finalista>/runs/seed_*.json` e são puladas. Cada
finalista fica preso à máquina, aos dados e às versões em que começou
(`fingerprint.json`): para trocar de PC, use outra pasta de saída.

### Tempo esperado

Medido na RTX 4070 Laptop: 0,09–0,16 s por época (os modelos são pequenos; o
custo é dominado pelo overhead por lote, então a 5060 Ti não deve ser muito
mais rápida). Carregar os dados do zip leva ~10 s. No pior caso, com todas as
execuções indo até 1.000 épocas, são **≈ 2 h para as 60 execuções** (≈ 1 h
para as 40 publicadas). O early stopping costuma encurtar isso. Deixe o
laptop na tomada e sem suspensão.

A execução real de 29/10/2026 levou **2 h 25 min**: 0,38 h no CV, 0,72 h no FV
*same* e 1,32 h no FV *valid*. No CV, a época levou o mesmo tempo da medição
(0,11 s). Nos dois FV levou cerca do dobro (0,18 e 0,30 s), com um TPE
rodando em paralelo na CPU. O laço de treino depende da CPU. O braço Keras
(30/10/2026) levou **41 min** para 40 execuções: 21,0 min no CV e 20,2 min no
FV.

Num teste de 250 épocas, o CV publicado ficou preso na perda de acaso
(ln 10 = 2,3026; 10%) com a seed 1009 e aprendeu com 1013, 1019 e 1021. Logo,
colapsos podem aparecer na distribuição de 20 seeds, como já acontece no F25
PAA-FV ajustado (3 execuções em 10%, 2 delas com todo o teste numa única
classe). As colunas `chance_level_runs` (acurácia ≤ 10%) e `single_class_runs`
(todo o teste numa classe) contam esses casos; `f25_tuned_chance_level_runs`
dá o mesmo para o F25 ajustado.

## Saídas

```
results/f25_published_finalists/
  summary.csv                    uma linha por finalista
  data_check.json                re-pontuação dos checkpoints F25
  <finalista>/metrics.json       mesmo formato do metrics.json do relatório F25
  <finalista>/runs/seed_*.json   um arquivo por seed (retomada)
  <finalista>/fingerprint.json   máquina/dados/versões da execução
  full_run.log                   console de 29/10 (as 3 finalistas PyTorch)
  keras_run.log                  console de 30/10 (braço Keras; as 60 anteriores reaproveitadas)
  smoke/                         teste de encadeamento (3 épocas), não é resultado
```

O `metrics.json` mantém as chaves do F25 (`results[]` com `seed`, `best_epoch`,
`validation_loss`, `validation_accuracy`, `test_loss`, `parameter_count`,
`flattened_units` e `metrics` com acurácias e matriz de confusão) e acrescenta
`epochs_run`, `wall_seconds`, `summary` (média, DP amostral, mín., máx.,
mediana), `comparison` (publicado e F25 ajustado), proveniência e ambiente.
O `summary.csv` traz as acurácias como frações. O `.gitignore` abre exceção
para ele, para os dois logs e para os CSVs de `results/f25/`, que as regras
`*.csv` e `*.log` deixariam de fora; a pasta `smoke/` fica fora do git.

## Ressalvas

- O braço Keras testa só a inicialização e o Adam do Keras. As outras
  diferenças do TensorFlow/Keras (ReduceLROnPlateau e EarlyStopping do Keras,
  TF32) continuam dentro do termo de dados (linha `learning_framework` da
  tabela de desvios do contrato).
- A GPU difere da RTX 2060 do F25. Os números de cada GPU diferem no
  arredondamento, o que a regra do F25 permite entre jobs; dentro de um
  finalista, todas as seeds rodam na mesma máquina.
- Nunca adicionar o `f25.zip` ao git.
