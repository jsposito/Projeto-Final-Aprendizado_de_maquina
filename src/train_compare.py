import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.preprocess import preprocessing_pipeline


# ============================================================
# CONFIGURAÇÕES
# ============================================================
TRACKING_URI = "http://localhost:5000/"
EXPERIMENT_NAME = "Monitoramento_Ambiental_Comparacao_Modelos_V2"
TARGET = "Qualidade_Ambiental"

RAW_DATA_PATH = "data/raw/dataset_ambiental.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Evita erro de run ativa
while mlflow.active_run() is not None:
    mlflow.end_run()


# ============================================================
# LEITURA E PRÉ-PROCESSAMENTO
# ============================================================
df = pd.read_csv(RAW_DATA_PATH)
df = preprocessing_pipeline(df, output_path=PROCESSED_DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 60% treino, 20% validação, 20% teste
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)


# ============================================================
# MODELOS
# ============================================================
models = {
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        ))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            random_state=42
        ))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=22,
            p=2,
            weights="distance"
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=100,
            gamma="scale",
            kernel="linear"
        ))
    ]),
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=100,
            max_iter=3000,
            solver="lbfgs"
        ))
    ]),
}


# ============================================================
# TREINO E AVALIAÇÃO
# ============================================================
results = []
best_model = None
best_model_name = None
best_score = -1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, pipeline in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        pred_train = pipeline.predict(X_train)
        pred_val = pipeline.predict(X_val)
        pred_test = pipeline.predict(X_test)

        scores = cross_validate(
            pipeline,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring={
                "f1_weighted": "f1_weighted",
                "accuracy": "accuracy"
            },
            return_train_score=False,
            n_jobs=None
        )

        row = {
            "modelo": model_name,
            "acc_train": accuracy_score(y_train, pred_train),
            "acc_val": accuracy_score(y_val, pred_val),
            "acc_test": accuracy_score(y_test, pred_test),
            "f1_train": f1_score(y_train, pred_train, average="weighted", zero_division=0),
            "f1_val": f1_score(y_val, pred_val, average="weighted", zero_division=0),
            "f1_test": f1_score(y_test, pred_test, average="weighted", zero_division=0),
            "precision_val": precision_score(y_val, pred_val, average="weighted", zero_division=0),
            "recall_val": recall_score(y_val, pred_val, average="weighted", zero_division=0),
            "f1_cv": scores["test_f1_weighted"].mean(),
            "params": str(pipeline.get_params())
        }

        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics({
            "acc_train": row["acc_train"],
            "acc_val": row["acc_val"],
            "acc_test": row["acc_test"],
            "f1_train": row["f1_train"],
            "f1_val": row["f1_val"],
            "f1_test": row["f1_test"],
            "precision_val": row["precision_val"],
            "recall_val": row["recall_val"],
            "f1_cv": row["f1_cv"]
        })

        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=model_name
            )
        except Exception as e:
            print(f"Aviso ao salvar {model_name} no MLflow: {e}")

        results.append(row)

        # Critério final: melhor f1 de validação
        if row["f1_val"] > best_score:
            best_score = row["f1_val"]
            best_model = pipeline
            best_model_name = model_name


# ============================================================
# CONSOLIDAÇÃO DOS RESULTADOS
# ============================================================
results_df = pd.DataFrame(results)

results_df["rank_f1_val"] = results_df["f1_val"].rank(ascending=False, method="min")
results_df["rank_acc_val"] = results_df["acc_val"].rank(ascending=False, method="min")
results_df["rank_f1_cv"] = results_df["f1_cv"].rank(ascending=False, method="min")
results_df["score_total"] = (
    results_df["rank_f1_val"] +
    results_df["rank_acc_val"] +
    results_df["rank_f1_cv"]
)

results_df = results_df.sort_values("score_total", ascending=True)

results_path = os.path.join(ARTIFACTS_DIR, "comparacao_modelos_v2.csv")
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
joblib.dump(best_model, best_model_path)

print("\nResultados completos:")
print(results_df[[
    "modelo",
    "acc_train", "acc_val", "acc_test",
    "f1_train", "f1_val", "f1_test",
    "precision_val", "recall_val", "f1_cv", "params"
]])

print("\n🏆 Ranking dos modelos:")
print(results_df[["modelo", "rank_f1_val", "rank_acc_val", "rank_f1_cv", "score_total"]])

print(f"\n🥇 Melhor modelo: {best_model_name}")
print(f"Melhor score (f1_val): {best_score:.6f}")
print(f"\nArquivo salvo em: {results_path}")
print(f"Melhor modelo salvo em: {best_model_path}")
print("Abra o MLflow em: http://localhost:5000/")