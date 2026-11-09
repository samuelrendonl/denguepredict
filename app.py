from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from src.modeling import (
    BINARY_INPUT_COLUMNS,
    FEATURE_LABELS,
    INPUT_OPTIONS,
    INPUT_FEATURE_COLUMNS,
    MODEL_LABELS,
    TARGET_COLUMN,
    batch_metrics,
    load_dataset,
    load_saved_bundle,
    normalize_columns,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "MASTER_CHART_F1000Research.xlsx"
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.config["SECRET_KEY"] = "dengue-analitica-academica"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def initialize_application() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATASET_PATH}")

    dataset = load_dataset(DATASET_PATH)
    app.config["DATASET"] = dataset
    # Los modelos deben ser generados desde los notebooks del proyecto.
    app.config["BUNDLE"] = load_saved_bundle(MODELS_DIR, dataset)
    app.config["DOWNLOADS"] = {}


initialize_application()


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/D"
    return f"{value * 100:.1f}%"


def metrics_for_template(metrics: dict) -> dict:
    result = dict(metrics)
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        result[f"{key}_display"] = format_percent(metrics.get(key))
    return result


@app.template_filter("percent")
def percent_filter(value):
    return format_percent(value)


@app.context_processor
def inject_globals():
    return {
        "binary_input_columns": BINARY_INPUT_COLUMNS,
        "input_options": INPUT_OPTIONS,
        "model_labels": MODEL_LABELS,
        "feature_labels": FEATURE_LABELS,
        "target_column": TARGET_COLUMN,
    }


@app.route("/")
def index():
    bundle = app.config["BUNDLE"]
    dataset = app.config["DATASET"]
    return render_template(
        "index.html",
        dataset_rows=len(dataset),
        complicated=int((dataset[TARGET_COLUMN] == 2).sum()),
        uncomplicated=int((dataset[TARGET_COLUMN] == 1).sum()),
        metrics={name: metrics_for_template(value) for name, value in bundle.metrics.items()},
        train_rows=bundle.train_rows,
        test_rows=bundle.test_rows,
    )


@app.route("/individual", methods=["GET", "POST"])
def individual():
    prediction_results = None
    form_values = {}

    if request.method == "POST":
        try:
            form_values = request.form.to_dict()
            model_name = request.form.get("model", "both")
            missing = [
                column for column in INPUT_FEATURE_COLUMNS
                if not request.form.get(column, "").strip()
            ]
            if missing:
                raise ValueError(
                    "Completa todas las variables requeridas: " + ", ".join(missing)
                )

            row = {
                column: (
                    int(float(request.form[column]))
                    if column in BINARY_INPUT_COLUMNS
                    else float(request.form[column])
                )
                for column in INPUT_FEATURE_COLUMNS
            }
            patient = pd.DataFrame([row], columns=INPUT_FEATURE_COLUMNS)
            bundle = app.config["BUNDLE"]
            names = list(MODEL_LABELS) if model_name == "both" else [model_name]
            prediction_results = []

            for name in names:
                predictions, probabilities = bundle.predict(patient, name)
                prediction_results.append(
                    {
                        "name": name,
                        "label": MODEL_LABELS[name],
                        "prediction": "Dengue complicado" if predictions[0] else "Dengue no complicado",
                        "probability": float(probabilities[0]),
                        "positive": bool(predictions[0]),
                    }
                )
        except (KeyError, TypeError, ValueError) as error:
            flash(f"Revisa los datos ingresados: {error}", "danger")

    return render_template("individual.html", results=prediction_results, form_values=form_values)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    result = None
    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            flash("Selecciona un archivo CSV o Excel.", "danger")
            return render_template("batch.html")

        try:
            filename = uploaded.filename.lower()
            if filename.endswith(".csv"):
                data = pd.read_csv(uploaded)
            elif filename.endswith((".xlsx", ".xls")):
                data = pd.read_excel(uploaded)
            else:
                raise ValueError("El archivo debe tener formato CSV o Excel.")

            data = normalize_columns(data)
            bundle = app.config["BUNDLE"]
            predictions = bundle.predict_all(data)
            metric_results = {
                name: batch_metrics(bundle, data, name) for name in MODEL_LABELS
            }
            result = {
                "rows": len(data),
                "columns": list(data.columns),
                "preview": predictions.head(12).to_dict(orient="records"),
                "preview_columns": list(predictions.columns),
                "metrics": {
                    name: metrics_for_template(value) if value else None
                    for name, value in metric_results.items()
                },
            }

            token = str(uuid.uuid4())
            app.config["DOWNLOADS"][token] = predictions
            result["download_token"] = token
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
            flash(f"No fue posible procesar el archivo: {error}", "danger")

    return render_template("batch.html", result=result)


@app.route("/download/<token>")
def download(token: str):
    data = app.config["DOWNLOADS"].get(token)
    if data is None:
        flash("El resultado ya no está disponible. Vuelve a procesar el archivo.", "warning")
        return redirect(url_for("batch"))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="Predicciones")
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="predicciones_dengue.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
