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
