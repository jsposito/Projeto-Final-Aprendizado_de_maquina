import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import preprocessing_pipeline


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
TRACKING_URI = "http://localhost:5000/"
EXPERIMENT_NAME = "Monitoramento_Ambiental_Comparacao_Modelos"
TARGET = "Qualidade_Ambiental"

RAW_DATA_PATH = "data/raw/dataset_ambiental.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Evita erro de run já ativa no Jupyter
while mlflow.active_run() is not None:
    mlflow.end_run()


# ============================================================
# 2. LEITURA E PRÉ-PROCESSAMENTO
# ============================================================
df = pd.read_csv(RAW_DATA_PATH)
df = preprocessing_pipeline(df, output_path="data/processed/processed_data.csv")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. MODELOS A COMPARAR
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
}

results = []
best_model = None
best_model_name = None
best_f1 = -1


# ============================================================
# 4. TREINO E COMPARAÇÃO
# ============================================================
for model_name, model in models.items():
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        }

        mlflow.log_param("model_name", model_name)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        mlflow.log_metrics(metrics)

        # Salvar modelo localmente
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Tentar salvar também no formato MLflow
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name
            )
        except Exception as e:
            print(f"Aviso: não foi possível salvar {model_name} no formato MLflow: {e}")

        results.append({
            "modelo": model_name,
            **metrics
        })

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_model = model
            best_model_name = model_name


# ============================================================
# 5. RESULTADO FINAL
# ============================================================
results_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
results_path = os.path.join(ARTIFACTS_DIR, "comparacao_modelos.csv")
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

print("\nComparação dos modelos:")
print(results_df)

print(f"\nMelhor modelo: {best_model_name}")
print(f"Melhor F1-Weighted: {best_f1:.4f}")

# Salvar melhor modelo separadamente
best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
joblib.dump(best_model, best_model_path)

print(f"\nArquivo salvo em: {results_path}")
print(f"Melhor modelo salvo em: {best_model_path}")
print("Abra o MLflow em: http://localhost:5000/")