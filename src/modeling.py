"""Carga, entrenamiento y evaluación de los modelos de dengue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


TARGET_COLUMN = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"
INPUT_FEATURE_COLUMNS = [
    "AGE",
    "GENDER",
    "DURATION OF FEVER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY",
    "HB",
    "PCV",
    "LOWEST WBC",
    "PLATELET COUNT",
    "HIGHEST WBC",
    "ALT",
    "CREATININE",
    "FERRITIN",
    "IV FLUIDS",
    "ANTIBIOTICS",
    "ANTIPYRETICS",
    "STEROIDS",
]

BINARY_INPUT_COLUMNS = [
    "AGE",
    "GENDER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY",
    "IV FLUIDS",
    "ANTIBIOTICS",
    "ANTIPYRETICS",
    "STEROIDS",
]

INPUT_OPTIONS = {
    "AGE": [("1", "6 años o menos"), ("2", "Más de 6 años")],
    "GENDER": [("1", "Masculino"), ("2", "Femenino")],
    "HEADACHE": [("1", "Sí"), ("2", "No")],
    "MYALGIA": [("1", "Sí"), ("2", "No")],
    "ABDOMINAL PAIN": [("1", "Sí"), ("2", "No")],
    "RASH": [("1", "Sí"), ("2", "No")],
    "VOMITING": [("1", "Sí"), ("2", "No")],
    "BREATHLESSNESS": [("1", "Sí"), ("2", "No")],
    "BLEEDING": [("1", "Sí"), ("2", "No")],
    "BP AT ADMISSION": [("2", "Sí, presenta hipotensión"), ("1", "No presenta hipotensión")],
    "ORGANOMEGALY": [("1", "Sí"), ("2", "No")],
    "IV FLUIDS": [("1", "Sí"), ("2", "No")],
    "ANTIBIOTICS": [("1", "Sí"), ("2", "No")],
    "ANTIPYRETICS": [("1", "Sí"), ("2", "No")],
    "STEROIDS": [("1", "Sí"), ("2", "No")],
}

# Alias de compatibilidad: son las columnas que el usuario entrega a la aplicación.
FEATURE_COLUMNS = INPUT_FEATURE_COLUMNS

MODEL_FEATURE_COLUMNS = [
    *INPUT_FEATURE_COLUMNS,
    "PCV_HIGH",
    "PLATELETS_LOW",
    "WBC_HIGH",
    "ALT_ELEVATED",
    "HYPOTENSION",
    "FEVER_LONG",
    "THROMBOCYTOPENIA_SEVERE",
    "PLATELET_PCV_RATIO",
]

FEATURE_LABELS = {
    "AGE": "Edad",
    "GENDER": "Género",
    "DURATION OF FEVER": "Duración de fiebre (días)",
    "HEADACHE": "Dolor de cabeza",
    "MYALGIA": "Mialgia",
    "ABDOMINAL PAIN": "Dolor abdominal",
    "RASH": "Erupción cutánea",
    "VOMITING": "Vómito",
    "BREATHLESSNESS": "Dificultad respiratoria",
    "BLEEDING": "Sangrado",
    "BP AT ADMISSION": "¿Presenta hipotensión al ingreso?",
    "ORGANOMEGALY": "Organomegalia",
    "HB": "Hemoglobina (g/dL)",
    "PCV": "PCV (%)",
    "LOWEST WBC": "WBC mínimo (células/mm³)",
    "PLATELET COUNT": "Plaquetas (células/mm³)",
    "HIGHEST WBC": "WBC máximo (células/mm³)",
    "ALT": "ALT (U/L)",
    "CREATININE": "Creatinina (mg/dL)",
    "FERRITIN": "Ferritina",
    "IV FLUIDS": "¿Recibe líquidos intravenosos?",
    "ANTIBIOTICS": "¿Recibe antibióticos?",
    "ANTIPYRETICS": "¿Recibe antipiréticos?",
    "STEROIDS": "¿Recibe esteroides?",
}

MODEL_LABELS = {
    "logistic": "Regresión Logística",
    "neural": "Red Neuronal Artificial",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina espacios innecesarios de los encabezados del archivo."""

    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga el dataset original desde Excel."""

    return normalize_columns(pd.read_excel(path))


def _threshold_feature(series: pd.Series, threshold: float, operator: str) -> pd.Series:
    """Crea una variable binaria conservando NaN cuando falta el valor original."""

    if operator == ">=":
        values = series >= threshold
    elif operator == "<=":
        values = series <= threshold
    elif operator == "<":
        values = series < threshold
    else:
        raise ValueError(f"Operador no soportado: {operator}")
    return pd.Series(np.where(series.isna(), np.nan, values.astype(float)), index=series.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia las entradas y crea las variables derivadas usadas por los modelos."""

    df = normalize_columns(df)
    missing = [column for column in INPUT_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))

    raw = df[INPUT_FEATURE_COLUMNS].copy()
    for column in INPUT_FEATURE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # Se conservan todas las variables originales del dataset. Las variables
    # derivadas se agregan además para mantener las transformaciones usadas
    # por el modelo.
    features = raw[INPUT_FEATURE_COLUMNS].copy()
    features["PCV_HIGH"] = _threshold_feature(raw["PCV"], 42, ">=")
    features["PLATELETS_LOW"] = _threshold_feature(raw["PLATELET COUNT"], 75000, "<=")
    features["WBC_HIGH"] = _threshold_feature(raw["HIGHEST WBC"], 7000, ">=")
    features["ALT_ELEVATED"] = _threshold_feature(raw["ALT"], 70, ">=")

    # En el dataset: BP AT ADMISSION = 2 sí tiene hipotensión, 1 no.
    bp = raw["BP AT ADMISSION"]
    features["HYPOTENSION"] = pd.Series(
        np.where(bp.isna(), np.nan, (bp == 2).astype(float)), index=raw.index
    )
    features["FEVER_LONG"] = _threshold_feature(raw["DURATION OF FEVER"], 6, ">=")
    features["THROMBOCYTOPENIA_SEVERE"] = _threshold_feature(
        raw["PLATELET COUNT"], 50000, "<"
    )
    features["PLATELET_PCV_RATIO"] = raw["PLATELET COUNT"] / raw["PCV"]

    for column in MODEL_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features[MODEL_FEATURE_COLUMNS]


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre público para reutilizar la misma limpieza en notebooks y Flask."""

    return engineer_features(df)


def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Elimina columnas con un único valor y devuelve sus nombres."""

    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    return df.drop(columns=constant_columns), constant_columns


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_model_data(df)


def _prepare_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"El archivo debe incluir la columna '{TARGET_COLUMN}'.")

    raw_target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if raw_target.isna().any() or not raw_target.isin([1, 2]).all():
        raise ValueError("La variable objetivo solo puede contener 1 o 2.")

    # El dataset original codifica 1 = no complicado y 2 = complicado.
    return (raw_target == 2).astype(int)


def build_models() -> dict[str, Pipeline]:
    """Crea los dos pipelines solicitados por la actividad."""

    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    return {
        "logistic": Pipeline(
            preprocessing
            + [
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                )
            ]
        ),
        "neural": Pipeline(
            preprocessing
            + [
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.0005,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=30,
                        max_iter=1500,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def calculate_metrics(y_true: pd.Series | np.ndarray, predictions: Any, probabilities: Any) -> dict[str, Any]:
    """Calcula las métricas binarias con 1 como clase positiva."""

    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "support": int(len(y_true)),
    }


@dataclass
class ModelBundle:
    models: dict[str, Pipeline]
    metrics: dict[str, dict[str, Any]]
    feature_columns: list[str]
    train_rows: int
    test_rows: int

    def predict(self, data: pd.DataFrame, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        if model_name not in self.models:
            raise ValueError("Modelo no válido.")
        features = _prepare_features(normalize_columns(data))
        model = self.models[model_name]
        predictions = model.predict(features).astype(int)
        probabilities = model.predict_proba(features)[:, 1]
        return predictions, probabilities

    def predict_all(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for model_name in self.models:
            predictions, probabilities = self.predict(data, model_name)
            result[f"PREDICCION_{model_name.upper()}"] = np.where(
                predictions == 1, "Complicado", "No complicado"
            )
            result[f"PROBABILIDAD_{model_name.upper()}_COMPLICADO"] = np.round(
                probabilities, 4
            )
        return result


def train_bundle(dataset: pd.DataFrame) -> ModelBundle:
    """Entrena usando una partición estratificada y evalúa sobre datos no vistos."""

    dataset = normalize_columns(dataset)
    features = _prepare_features(dataset)
    target = _prepare_target(dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    models = build_models()
    metrics: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics[model_name] = calculate_metrics(y_test, predictions, probabilities)

    return ModelBundle(
        models=models,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def load_saved_bundle(models_dir: str | Path, dataset: pd.DataFrame) -> ModelBundle:
    """Carga los pipelines y la evaluación generados por el notebook de entrenamiento."""

    models_dir = Path(models_dir)
    logistic_path = models_dir / "modelo_logistico.joblib"
    neural_path = models_dir / "modelo_neuronal.joblib"
    evaluation_path = models_dir / "evaluation_data.joblib"

    required_files = [logistic_path, neural_path, evaluation_path]
    missing = [str(path.name) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "No se encontraron los modelos entrenados por el notebook: "
            + ", ".join(missing)
            + ". Ejecuta notebooks/02_entrenamiento_modelos.ipynb primero."
        )

    models = {
        "logistic": joblib.load(logistic_path),
        "neural": joblib.load(neural_path),
    }
    evaluation = joblib.load(evaluation_path)
    metrics = {}
    for model_name, model in models.items():
        predictions = model.predict(evaluation["x_test"])
        probabilities = model.predict_proba(evaluation["x_test"])[:, 1]
        metrics[model_name] = calculate_metrics(
            evaluation["y_test"], predictions, probabilities
        )

    return ModelBundle(
        models=models,
        metrics=metrics,
        feature_columns=list(evaluation.get("feature_columns", MODEL_FEATURE_COLUMNS)),
        train_rows=int(evaluation["train_rows"]),
        test_rows=int(evaluation["test_rows"]),
    )


def batch_metrics(bundle: ModelBundle, data: pd.DataFrame, model_name: str) -> dict[str, Any] | None:
    """Calcula métricas de un lote cuando este contiene la columna objetivo."""

    data = normalize_columns(data)
    if TARGET_COLUMN not in data.columns:
        return None

    target = _prepare_target(data)
    predictions, probabilities = bundle.predict(data, model_name)
    return calculate_metrics(target, predictions, probabilities)
