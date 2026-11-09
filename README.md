# DenguePredict

Aplicación web académica en Flask para predecir el riesgo de dengue complicado en pacientes pediátricos usando:

- Regresión Logística.
- Red Neuronal Artificial (`MLPClassifier`).

## Dataset

La aplicación utiliza `MASTER_CHART_F1000Research.xlsx`. La variable objetivo está codificada así:

- `1`: dengue no complicado.
- `2`: dengue complicado.

El modelo utiliza todas las variables de entrada disponibles en el dataset y variables derivadas de limpieza:

- `AGE`.
- `GENDER`.
- `DURATION OF FEVER`.
- `HEADACHE`.
- `MYALGIA`.
- `ABDOMINAL PAIN`.
- `RASH`.
- `VOMITING`.
- `BREATHLESSNESS`.
- `BLEEDING`.
- `ORGANOMEGALY`.
- `HB`.
- `PCV`.
- `LOWEST WBC`.
- `PLATELET COUNT`.
- `HIGHEST WBC`.
- `ALT`.
- `CREATININE`.
- `FERRITIN`.
- `IV FLUIDS`.
- `ANTIBIOTICS`.
- `ANTIPYRETICS`.
- `STEROIDS`.
- `BP AT ADMISSION` (2 = hipotensión, 1 = no hipotensión).

También se crean `PCV_HIGH`, `PLATELETS_LOW`, `WBC_HIGH`, `ALT_ELEVATED`, `HYPOTENSION`, `FEVER_LONG`, `THROMBOCYTOPENIA_SEVERE` y `PLATELET_PCV_RATIO`. La variable objetivo no se solicita al predecir: es el resultado que calcula el modelo.

Los notebooks documentan la conversión de códigos 1/2 a valores numéricos y la creación de variables nuevas. Los valores faltantes se completan con la mediana dentro del pipeline de cada modelo.

## Instalación y ejecución

Se recomienda crear un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python -m jupyter lab
```

Ejecuta los notebooks en este orden:

1. `notebooks/01_eda_dengue.ipynb` para la exploración.
2. `notebooks/02_entrenamiento_modelos.ipynb` para generar los modelos en `models/`.
3. `notebooks/03_evaluacion_modelos.ipynb` para métricas, matrices de confusión y curvas ROC.

Después inicia la aplicación web:

```bash
python app.py
```

Luego abrir `http://127.0.0.1:5000`.

## Publicación en Render

El proyecto incluye `Procfile` y `render.yaml`. Para obtener el enlace público:

1. Sube este proyecto a un repositorio de GitHub, incluyendo la carpeta `models/` generada por el notebook de entrenamiento.
2. En Render selecciona **New + → Web Service** y conecta el repositorio.
3. Usa `Build Command: pip install -r requirements.txt` y `Start Command: gunicorn app:app`.
4. Espera el despliegue y verifica la URL `*.onrender.com` antes de entregar.

Si se vuelven a entrenar los modelos, ejecuta nuevamente el notebook 02 y sube los archivos actualizados de `models/`.

## Funcionalidades

### Predicción individual

Permite introducir los datos de un paciente y obtener la clase predicha y la probabilidad de complicación con uno o ambos modelos.

### Predicción por lote

Permite cargar CSV, XLSX o XLS. El archivo debe incluir las 24 variables de entrada del dataset; las variables derivadas se crean automáticamente. Si además incluye `GROUPS: DENGUE FEVER OR COMPLICATED DENGUE`, se calculan accuracy, precision, sensibilidad, especificidad, F1-score, ROC-AUC y matriz de confusión. También se puede descargar el resultado en Excel.

En `ejemplos/` se incluyen `pacientes_ejemplo.csv` para probar predicciones y `pacientes_ejemplo_con_etiquetas.csv` para probar también las métricas.

## Evaluación

Al iniciar, los datos se dividen de forma estratificada en 80% para entrenamiento y 20% para prueba. Las métricas de la página de inicio corresponden a ese conjunto de prueba no utilizado durante el entrenamiento.

> Esta aplicación es un proyecto académico y no sustituye el diagnóstico ni la valoración de un profesional de la salud.
