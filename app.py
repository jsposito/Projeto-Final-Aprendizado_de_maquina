import joblib
import gradio as gr
import pandas as pd

# Carrega o melhor modelo
modelo = joblib.load("artifacts/models/best_model.joblib")

# Ordem das variáveis esperadas pelo modelo
FEATURES = [
    "Temperatura",
    "Umidade",
    "CO2",
    "CO",
    "Pressao_Atm",
    "NO2",
    "SO2",
    "O3"
]

def prever_qualidade(temperatura, umidade, co2, co, pressao_atm, no2, so2, o3):
    entrada = pd.DataFrame([[
        temperatura, umidade, co2, co, pressao_atm, no2, so2, o3
    ]], columns=FEATURES)

    previsao = modelo.predict(entrada)[0]

    return (
        f"Qualidade Ambiental Prevista: {previsao}\n\n"
        "Este conteúdo é destinado apenas para fins educacionais. "
        "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
    )

app = gr.Interface(
    fn=prever_qualidade,
    inputs=[
        gr.Number(label="Temperatura"),
        gr.Number(label="Umidade"),
        gr.Number(label="CO2"),
        gr.Number(label="CO"),
        gr.Number(label="Pressão Atmosférica"),
        gr.Number(label="NO2"),
        gr.Number(label="SO2"),
        gr.Number(label="O3"),
    ],
    outputs=gr.Textbox(label="Resultado"),
    title="Classificação da Qualidade Ambiental",
    description="Aplicação para prever a qualidade ambiental com base em dados de sensores."
)

if __name__ == "__main__":
    app.launch()