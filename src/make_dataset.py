import pandas as pd
from src.preprocess import preprocessing_pipeline

df = pd.read_csv("data/raw/dataset_ambiental.csv")
df_processado = preprocessing_pipeline(df)

print(df_processado.head())