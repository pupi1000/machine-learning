readme_content = """# Sistema Integrado de Predicción de Movilidad Urbana - Grupo 5
## Asignatura: Machine Learning (2026-I)
### Metodología Obligatoria: CRISP-DM

Este repositorio contiene la solución completa y articulada para la gestión de demanda de bicicletas basado en el **Bike Sharing Dataset (UCI)**.

## 🚀 Estructura del Pipeline Operativo
* **`/data`**: Contiene el dataset histórico origen `hour.csv`.
* **`/notebooks`**: Contiene los cuadernos experimentales de las Fases 2, 3 y 4.
* **`/models`**: Almacena los artefactos serializados (`.pkl`) validados por MLflow.
* **`/predictions_store`**: Repositorio de persistencia JSON de la inferencia periódica continua.
* **`/metrics_history`**: Registro histórico de auditoría de telemetría y alertas de Data Drift.

## 🛠️ Instrucciones de Ejecución Reproducible
1. **Fase 2 (ML Clásico):** Ejecutar para observar la limpieza de Data Leakage y el benchmark de AutoML (XGBoost Campeón con MAE 21.58).
2. **Fase 3 (Deep Learning):** Inicializa la arquitectura de red recurrente LSTM en PyTorch bajo ventanas secuenciales de 24 horas y extrae el Espacio Latente (Embeddings).
3. **Fase 4 (MLOps + RAG):** Inicializa el motor de tracking MLflow, ejecuta el cron-job de inferencia periódica, activa el Monitoreo de Deriva Temporal y habilita el Motor de Búsqueda Vectorial por Distancia Euclidiana para recuperar días similares del pasado.

## 👥 Integrantes del Grupo 5
* Einar Guillen
* Octavio
* Leandro
* Huascar
* Leonardo


