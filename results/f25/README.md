# Análise da campanha F25 — RTX 2060

Importação e análise: 29 de outubro de 2026. Fonte: `f25.zip` na raiz do
repositório. SHA-256 do arquivo:
`684128d76b447b255b31ce5ba276f26f50976313abd433863036bf7df3fcacaa`.
Revisão independente (Claude + verificador Fable 5.1) no mesmo dia: todos os
números abaixo foram recalculados das 1.700 matrizes de confusão, das 22 bases
Optuna e dos registros de execução.

## Resultado principal

A campanha devolvida está completa: **22 estudos HPO, 2.200 propostas completas
com cinco execuções cada e 85 configurações com 20 seeds de avaliação**,
totalizando 1.700 avaliações. Os 12.700 ajustes do desenho retido não incluem
trabalho abortado ou repetido e não estimam tempo de GPU.

| Etapa | Configuração com maior média observada | Acurácia média ± DP |
|---|---|---:|
| F25-R, reconstrução | PAA, aceleração vertical da caixa (CV) | 57,33% ± 6,78 pp |
| F25-X, hiperparâmetros compartilhados | RAW, arfagem da caixa (CP) | 72,61% ± 4,95 pp |
| F25-X, ajuste por canal | PAA-MR, arfagem da caixa (CP) | 77,63% ± 2,24 pp |
| F25-X, pares (hiperparâmetros compartilhados) | PAA, CV+CP | 75,68% ± 4,59 pp |

DP = desvio padrão amostral entre 20 inicializações, em pontos percentuais.
Códigos do artigo: CV=S00, FV=S01, RV=S02, CP=S05, FP=S06, RP=S07 (os CSVs
mantêm os códigos do arquivo). Os valores descrevem um único catálogo simulado,
perfil e split; as melhores configurações foram identificadas após a leitura
do teste.

## O que os resultados sustentam

- **CP é a resposta mais informativa:** maior mediana nas seis combinações
  braço × estratégia de ajuste e maior média em cinco (a exceção é PAA-CNN com
  ajuste por canal, por causa de runs baixos de CP). FP e RP ficam perto do
  acaso (10%). Leitura física: a arfagem da caixa mede a diferença entre os
  truques, a 16 m, base comparável aos vãos de 19,95 m; a do truque usa só
  2,3 m (coerente com McGeown et al. 2024).
- **PAA-MR/CP ajustado:** 77,63% ± 2,24 pp, mediana 77,95%, sem run colapsado,
  43.642 parâmetros (contra 1.820.650 do PAA-CNN/CP e 7.497.738 do RAW-CNN/CP).
  A vantagem é sobretudo de **estabilidade**: pareado por seed, supera o
  PAA-CNN/CP em +10,37 pp na média mas só +1,45 pp na mediana (12/20 seeds).
  Os vencedores com flatten bateram no limite de 1.000 épocas (19/20 e 18/20
  runs com checkpoint ≥ 950), contra 3/20 do PAA-MR.
- **Ajuste nem sempre ajuda:** PAA-MR/CP +11,65 pp (DP pareado 3,64; 20/20
  seeds); PAA-CNN/CP −5,03 pp e RAW-CNN/RP −7,25 pp na média.
- **Otimismo da seleção:** o objetivo de validação do vencedor supera a média
  de teste em 10,5 pp em média (1,4–26,1) nas 22 configurações ajustadas; não é
  diferença validação/teste (nos runs de relatório elas diferem ≤ 4,2 pp), e
  sim seleção do melhor de 100 propostas em 200 passagens de validação.
- **Pares não dão ganho consistente:** o par PAA CV+CP (75,68%) só supera o
  PAA/CP isolado (72,29%) por causa de **um** run colapsado do isolado: as
  medianas são iguais (74,55%) e o par vence em 9/20 seeds pareados. Com
  RAW-CNN e PAA-MR, 11 de 15 pares ficam abaixo do melhor membro; a exceção
  consistente é PAA-MR RV+CP (+4,65 pp, 15/20).
- **Fissura limita a acurácia:** eixos 99,97% (rolamento), 97,63% (scour) e
  77,76% (fissura); maioria 60/40/50%. As oito confusões > 10% da matriz média
  envolvem pares de cenários que diferem só na fissura (Healthy/DC2, DC4/DC7,
  DC5/DC8, DC9/DC10: 37,5% e 39,6%). DC3 e DC6, únicos na combinação scour ×
  rolamento, têm 89,8% e 99,1%.

## Reconstrução e instabilidade

As matrizes publicadas por Fernandes et al. (2025) são a **melhor de 20
execuções** (seção 5.1 da fonte), então o comparador justo é o nosso máximo:
RAW/CV 71,8% (publicado 65,1%), PAA/CV 66,6% (86,7%), RAW/FV 65,5% (82,2%),
PAA/FV 73,8% (82,1%). O benefício do PAA na caixa reaparece na média (57,33%
contra 49,74%), com magnitude menor; PAA/FV tem três runs em 10%. A confusão
DC9/DC10 da fonte se reproduz; os erros DC4→Healthy da matriz publicada do
truque não (acurácia de rolamento ≥ 99,3% em todo run não colapsado).
Malha, fissura, framework, split e resultado da otimização diferem; a fonte é
também inconsistente quanto a padding e à faixa da camada densa (finalista do
truque com 96 neurônios fora da faixa 16–64). Os finalistas publicados não
foram retreinados — é o experimento de controle mais barato para dividir a
lacuna.

Entre os 1.700 ajustes, 90 predizem uma única classe, 98 têm exatamente 10% e
132 têm ≤ 10%; 60 dos 90 colapsos usam FP ou RP. Nos canais informativos e seus
pares, só PAA-CNN colapsa (14/320, 13 com os hiperparâmetros compartilhados do
truque dianteiro; RAW-CNN 0/320, PAA-MR 0/280). Há 379 checkpoints no epoch
950 ou depois (53 no epoch 1.000). Todos os runs permanecem nas distribuições.

## Integridade e ambiente

O script verifica os hashes dos 525 arquivos listados nos manifestos, os
manifestos/planos, os dez MATs de estados, `case_info.mat`, os 107 registros,
85 vencedores de referência e 55 continuações; recalcula as quatro acurácias
das 1.700 matrizes 10×10 (100 casos por classe) e confere propostas, seeds,
objetivos e parâmetros dos vencedores nas 22 bases SQLite.

Geração: MATLAB R2021a Update 7. Treino: Windows, RTX 2060 6 GB, PyTorch
2.6.0+cu118, CUDA 11.8, algoritmos determinísticos. Driver 591.86 → 610.88
durante a campanha: 18 otimizações terminaram sob o driver antigo, uma
atravessou a troca (PAA-MR/RV, retomada em 65 propostas) e três rodaram sob o
novo. Como `run_f25.py` reexecuta todos os jobs a cada lançamento e o relatório
retreina os 20 seeds do zero, **todos os 1.700 runs reportados foram calculados
sob o driver 610.88**. Quatro otimizações foram interrompidas e retomadas
(gaps de 30–154 h). Os JSONs não retêm predições por passagem.

## Arquivos e reprodução

Execute na raiz: `python analyze_f25_results.py --archive f25.zip`
(NumPy, SciPy, Matplotlib; lê o ZIP diretamente e não altera os dados).

- `audit.json`, `configuration_summary.csv`, `seed_metrics.csv`,
  `hpo_summary.csv`, `tuning_differences.csv`, `mean_confusion_matrices.csv`.
- `paper1/tables/f25_*.tex`: cinco tabelas (médias arredondadas meio-para-cima
  a partir das razões exatas; inclui os hiperparâmetros selecionados). A sexta,
  `f25_published_finalists.tex`, vem do controle com as finalistas publicadas
  (`results/f25_published_finalists/`, `run_f25_published_finalists.py
  --paper-table`).
- `paper1/figures/f25_*.{pdf,png}`: boxplots dos singles, pares e matriz média.

Os CSVs desta pasta são versionados (exceção no `.gitignore` desde
31/10/2026); o `f25.zip` (1,08 GB) não.

## Artigo

Por decisão do autor (29/10/2026), `paper1/` agora é um **artigo independente
só do F25**, no formato de Fernandes et al. O desenho da campanha de regressão
de quatro blocos foi preservado sem alterações em `paper_regression/` para o
futuro artigo de regressão. Em 31/10/2026 entrou o controle com as finalistas
publicadas (29 páginas), e a disponibilidade de dados passou a ser "sob pedido
ao autor correspondente". Faltam apenas itens dos autores (agradecimentos,
contribuições, financiamento, periódico e título), marcados em vermelho no
PDF.
