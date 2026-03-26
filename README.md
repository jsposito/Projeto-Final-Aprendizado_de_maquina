# Projeto Final - Aprendizado de Máquina

## Classificação da Qualidade Ambiental com Machine Learning

Este projeto tem como objetivo desenvolver um pipeline completo de aprendizado de máquina para **classificação da qualidade ambiental** com base em dados de sensores ambientais. A solução contempla desde a análise exploratória dos dados até o treinamento, comparação de modelos, rastreamento de experimentos com MLflow e deploy da aplicação no Hugging Face Spaces.

## Objetivo

Construir um modelo de machine learning capaz de prever a variável **`Qualidade_Ambiental`** a partir das seguintes variáveis:

- Temperatura
- Umidade
- CO2
- CO
- Pressao_Atm
- NO2
- SO2
- O3

## Etapas do Projeto

### 1. Análise Exploratória dos Dados
Foi realizada uma análise exploratória para compreender a estrutura do conjunto de dados, identificar valores ausentes, possíveis inconsistências, distribuição das variáveis e comportamento da variável alvo.

### 2. Pré-processamento
A etapa de pré-processamento foi implementada em `src/preprocess.py` e contempla:

- conversão da coluna `Pressao_Atm` para formato numérico;
- tratamento de valores ausentes;
- remoção de outliers com base no método do IQR;
- agrupamento das classes da variável alvo para simplificação da modelagem.

Mapeamento aplicado em `Qualidade_Ambiental`:

- `Excelente` → `Boa`
- `Boa` → `Boa`
- `Moderada` → `Moderada`
- `Ruim` → `Ruim`
- `Muito Ruim` → `Ruim`

### 3. Treinamento e Comparação de Modelos
Foram treinados e comparados os seguintes modelos:

- Logistic Regression
- Decision Tree
- Random Forest

A comparação foi realizada com base nas métricas:

- Accuracy
- Precision Weighted
- Recall Weighted
- F1 Weighted
- F1 Macro

### 4. Rastreamento com MLflow
Os experimentos foram monitorados com o **MLflow 3.10.1**, permitindo registrar:

- parâmetros dos modelos;
- métricas de desempenho;
- artefatos gerados;
- comparação entre execuções;
- versões dos modelos treinados.

### 5. Seleção do Melhor Modelo
Após a comparação dos modelos, o algoritmo **Logistic Regression** apresentou o melhor desempenho geral, com os seguintes resultados:

- **Accuracy:** 0.9615
- **F1 Weighted:** 0.9616

Esse modelo foi selecionado como modelo final do projeto.

### 6. Deploy da Aplicação
A aplicação final foi disponibilizada no **Hugging Face Spaces** com interface construída em **Gradio**, permitindo ao usuário inserir os valores dos sensores e obter a previsão da qualidade ambiental.


## Estrutura do Projeto

```text
Projeto-Final-Aprendizado_de_maquina/
├── app.py
├── README.md
├── requirements.txt
├── artifacts/
│   └── models/
│       └── best_model.joblib
├── data/
│   ├── raw/
│   │   └── dataset_ambiental.csv
│   └── processed/
│       └── processed_data.csv
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
└── src/
    ├── preprocess.py
    ├── make_dataset.py
    └── train_compare.py

