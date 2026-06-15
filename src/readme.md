# 🚲 Sistema Integrado de Predicción de Movilidad Urbana — Grupo 5

Este repositorio contiene la solución de fin a fin para la gestión y predicción de la demanda de bicicletas compartidas, basado en el dataset histórico **Bike Sharing Dataset (UCI)**. La solución abarca desde el modelado clásico y deep learning hasta la construcción de una plataforma de producción interactiva bajo principios **MLOps**, **Recuperación Semántica (RAG en Espacio Latente)** y **Telemetría de Modelos**.

El desarrollo del proyecto se rige estrictamente bajo la metodología estándar **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

---

## 📈 Fases del Proyecto (CRISP-DM)

### 1. Comprensión del Negocio y Datos (Business & Data Understanding)
* **Objetivo:** Optimizar la distribución y disponibilidad de bicicletas públicas mediante modelos predictivos para reducir pérdidas operativas por escasez o sobreabastecimiento.
* **Exploración:** Análisis multivariable de factores temporales (estaciones, meses, horas, días hábiles) y meteorológicos (temperatura, sensación térmica, humedad, velocidad del viento) del dataset `hour.csv` (2011-2012).

### 2. Preparación de Datos y Modelado Clásico (Data Prep & Modeling)
* **Evitación de Data Leakage:** Segmentación rigurosa de secuencias temporales para evitar el filtrado de información futura.
* **Modelos Benchmark:** Evaluación comparativa mediante AutoML y modelos tradicionales. 
* **Modelo Campeón:** Algoritmo XGBoost con un error medio absoluto de **MAE = 21.58** bicicletas en el conjunto de prueba.

### 3. Aprendizaje Profundo con Redes Recurrentes (Deep Learning)
* **Arquitectura:** Red Neuronal Recurrente **LSTM** de 2 capas implementada en PyTorch (`input_size=12`, `hidden_size=64`, `output_size=1`).
* **lookback Window:** Configuración secuencial basada en una ventana deslizante de **24 horas** de contexto climatológico e histórico.
* **Extracción del Espacio Latente (Embeddings):** La capa oculta final del modelo LSTM extrae una representación matemática compacta de **64 dimensiones** que resume el perfil de comportamiento de cada día.

### 4. MLOps, Búsqueda Vectorial y Telemetría en Producción
* ** predictions_store (JSON Auditoría):** Ingesta y registro de lotes de predicciones en producción estructurados en archivos JSON con metadatos de auditoría temporal (`startDate`, `endDate`, `recordCount`, `avgMae`, `avgRmse`).
* **Motor de Recuperación Semántica (RAG):** Motor de búsqueda que localiza los 3 escenarios del pasado (homólogos) más similares al día de consulta calculando la distancia euclidiana en el espacio latente de embeddings de 64D.
* **Monitoreo de Data Drift (Deriva Temporal):** Monitoreo continuo del error absoluto diario frente a un umbral crítico de **35.0 MAE** calculado a partir de una media móvil de 7 días.

---

## 🚀 Características del Dashboard Interactivo

El panel de control interactivo está desarrollado en **Streamlit** y estilizado con una interfaz de diseño premium inspirada en React (Tailwind CSS, tipografía *Outfit*, gradientes vibrantes y micro-animaciones en hover).

1. **📊 Inferencia Interactiva por Lotes (Tab 1):**
   * Predicción secuencial en tiempo real basada en filtros de fecha.
   * KPI Cards interactivas (MAE, RMSE, muestras procesadas).
   * Exportación a CSV de las predicciones e ingesta al Predictions Store.
   * Visor y administrador interactivo de lotes registrados.

2. **🔍 Búsqueda de Escenarios Homólogos (Tab 2):**
   * Contraste en columnas del día de consulta frente a los 3 días más parecidos del pasado mediante la distancia en el espacio latente.
   * Gráficos comparativos del perfil de demanda y temperatura de las 24 horas del perfil.

3. **📈 Telemetría y Alertas de Data Drift (Tab 3):**
   * Panel de control de estabilidad: Banners dinámicos de estado (**🟢 Nominal: Estable** frente a **🚨 Alerta: Data Drift Activo**).
   * Velocímetro interactivo de deriva basado en la media móvil de error.
   * Selector interactivo de fecha que actualiza los KPIs y velocímetro a cualquier día del historial.
   * Sobreescritura y recálculo automático cronológico al registrar métricas para fechas duplicadas.
   * Botón de restauración del historial simulado a valores base nominales.

4. **🔮 Pronóstico y Simulación What-If (Tab 4):**
   * Proyección a futuro de 24, 48 o 72 horas partiendo de cualquier fecha.
   * Controles climatológicos interactivos (sliders de temperatura, humedad y estado del cielo) para simular impactos en la demanda bajo escenarios extremos o ideales.

---

## 📁 Estructura del Repositorio

```bash
├── data/
│   └── hour.csv                 # Dataset original de alquiler de bicicletas
├── metrics_history/
│   └── monitoreo_diario.json    # Historial de telemetría de monitoreo diario
├── notebooks/
│   ├── FASE_2 (4).ipynb         # EDA y modelado clásico XGBoost
│   ├── FAse_3 (1).ipynb         # Entrenamiento LSTM PyTorch y embeddings
│   ├── fase_4.ipynb             # MLflow, tracking y pruebas RAG
│   └── fae_5.ipynb              # Pruebas adicionales y validación
├── predictions_store/
│   └── log_pred_*.json          # Logs de inferencia por lotes registrados
├── src/
│   ├── app/
│   │   └── app.py               # Código principal del Dashboard (Streamlit)
│   ├── config/
│   │   └── config.json          # Parámetros del modelo y rutas del pipeline
│   ├── models/
│   │   ├── bike_lstm_v1.pth     # Pesos serializados de la red LSTM
│   │   ├── bike_model_v1.pkl    # Pipeline del modelo campeón
│   │   ├── scaler_features.pkl  # Escalador de variables exógenas
│   │   └── scaler_target.pkl    # Escalador de la variable objetivo
│   └── reports/                 # Recursos gráficos del dashboard
└── REQUIREMENTS.txt             # Dependencias del proyecto
```

---

## 🛠️ Instrucciones de Ejecución

### 1. Instalación de Dependencias
Asegúrese de contar con Python 3.8+ instalado. Instale las dependencias ejecutando:
```bash
pip install -r REQUIREMENTS.txt
```

### 2. Ejecutar la Aplicación Dashboard
Inicie el panel interactivo en su servidor local con:
```bash
python -m streamlit run src/app/app.py
```
La aplicación se abrirá por defecto en `http://localhost:8501`.

---

## 👥 Integrantes del Grupo 5
* **Einar Guillen**
* **Octavio Luna**
* **Leandro Colque**
* **Huascar Duran**
* **Leonardo Ibarra**

---
*Asignatura: Machine Learning (2026-I) - Facultad de Ingeniería de sistemas.*
