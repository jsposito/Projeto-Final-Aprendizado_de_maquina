import pandas as pd


# ============================================================
# 1. FUNÇÃO AUXILIAR PARA LIMITES DO IQR
# ============================================================
def iqr_bounds(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return low, high


# ============================================================
# 2. REMOÇÃO DE OUTLIERS
# ============================================================
def remove_outliers_iqr(df, column, category_col=None):
    if column not in df.columns:
        raise ValueError(
            f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}"
        )

    if category_col and category_col not in df.columns:
        raise ValueError(
            f"Coluna de categoria '{category_col}' não encontrada."
        )

    if category_col:
        result = pd.DataFrame()

        for category in df[category_col].dropna().unique():
            subset = df[df[category_col] == category]
            low, high = iqr_bounds(subset[column])
            mask = (subset[column] >= low) & (subset[column] <= high)
            result = pd.concat([result, subset[mask]])

        return result.reset_index(drop=True)

    low, high = iqr_bounds(df[column])
    mask = (df[column] >= low) & (df[column] <= high)
    return df[mask].reset_index(drop=True)


# ============================================================
# 3. TRATAMENTO DE VALORES NULOS
# ============================================================
def missing_values(df):
    df = df.copy()

    cols_num = df.select_dtypes(include="number").columns
    for col in cols_num:
        df[col] = df[col].fillna(df[col].median())

    return df


# ============================================================
# 4. MAPEAMENTO / AGRUPAMENTO DAS CLASSES
# ============================================================
def mapping_data(df):
    df = df.copy()

    mapping = {
        "Excelente": "Boa",
        "Boa": "Boa",
        "Moderada": "Moderada",
        "Ruim": "Ruim",
        "Muito Ruim": "Ruim"
    }

    df["Qualidade_Ambiental"] = df["Qualidade_Ambiental"].map(mapping)
    return df


# ============================================================
# 5. PIPELINE DE PRÉ-PROCESSAMENTO
# ============================================================
def preprocessing_pipeline(df, output_path="data/processed/processed_data.csv"):
    df = df.copy()

    if "Pressao_Atm" in df.columns:
        df["Pressao_Atm"] = pd.to_numeric(df["Pressao_Atm"], errors="coerce")

    df = missing_values(df)

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col, "Qualidade_Ambiental")

    df = mapping_data(df)

    import os
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(output_path, index=False)
    return df