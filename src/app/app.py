import os
import json
import datetime
import random
import pickle
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go


def clean_html(html_str):
    return "\n".join([line.strip() for line in html_str.split("\n")])


# ---------------------------------------------------------
# 1. Configuración de la Página y Estilos Visuales (Premium)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Demanda & Telemetría - Grupo 5",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo corporativo premium utilizando CSS personalizado inspirado en React/Tailwind CSS
st.markdown("""
    <style>
    /* Tipografía y estilos globales */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #f8fafc;
    }
    
    html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, input, button, select, textarea {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* Título principal minimalista premium */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #0f172a 30%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Barra lateral (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #334155;
    }
    
    /* Pestañas (Tabs) personalizadas estilo Segmented Control */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f1f5f9 !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 6px !important;
        border-bottom: none !important;
        width: 100% !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        padding: 10px 20px !important;
        background-color: transparent !important;
        border-radius: 10px !important;
        border: none !important;
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important;
        background-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Entradas y controles estilizados */
    div[data-baseweb="input"], select, div[role="listbox"] {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }
    
    /* Botones estilo React (Slate dark) */
    div.stButton > button {
        background: #0f172a !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        border: 1px solid #0f172a !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        background: #1e293b !important;
        border-color: #1e293b !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Ajustes menores de espaciado */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
    }

    /* Estilos Premium para KPI Cards */
    .kpis-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 20px;
        margin-bottom: 28px;
        margin-top: 10px;
    }
    @media (max-width: 768px) {
        .kpis-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr)) !important;
        }
    }

    .premium-card {
        background: linear-gradient(135deg, #ffffff 60%, #f8fafc 100%) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .premium-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 20px -8px rgba(99, 102, 241, 0.15), 0 4px 6px -2px rgba(99, 102, 241, 0.05) !important;
        border-color: #cbd5e1 !important;
    }
    
    .metric-value-indigo {
        font-size: 38px !important;
        font-weight: 850 !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(135deg, #0f172a 30%, #4f46e5 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        line-height: 1.1 !important;
        margin-top: 4px !important;
    }
    
    .metric-value-emerald {
        font-size: 38px !important;
        font-weight: 850 !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(135deg, #064e3b 30%, #10b981 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        line-height: 1.1 !important;
        margin-top: 4px !important;
    }

    .metric-value-rose {
        font-size: 38px !important;
        font-weight: 850 !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(135deg, #7f1d1d 30%, #f43f5e 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        line-height: 1.1 !important;
        margin-top: 4px !important;
    }

    .metric-value-violet {
        font-size: 38px !important;
        font-weight: 850 !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(135deg, #4c1d95 30%, #8b5cf6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        line-height: 1.1 !important;
        margin-top: 4px !important;
    }

    .metric-icon-container {
        padding: 16px !important;
        border-radius: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        transition: all 0.3s ease !important;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------
# 2. Rutas y Clases de Modelos
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "hour.csv")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
METRICS_PATH = os.path.join(BASE_DIR, "metrics_history", "monitoreo_diario.json")
PRED_STORE_DIR = os.path.join(BASE_DIR, "predictions_store")

# Definición del modelo LSTM exactamente como en FASE_3
class DemandaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DemandaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        embedding_latente = out[:, -1, :]
        prediccion = self.fc(embedding_latente)
        return prediccion, embedding_latente

# ---------------------------------------------------------
# 3. Funciones de Carga y Preprocesamiento
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['dteday_parsed'] = pd.to_datetime(df['dteday'])
    return df

@st.cache_resource
def load_ml_resources():
    with open(os.path.join(MODELS_DIR, "scaler_features.pkl"), "rb") as f:
        scaler_features = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler_target.pkl"), "rb") as f:
        scaler_target = pickle.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DemandaLSTM(input_size=12, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "bike_lstm_v1.pth"), map_location=device))
    model.to(device)
    model.eval()
    
    return model, scaler_features, scaler_target, device

@st.cache_resource
def precompute_embeddings(_model, _df, _scaler_features):
    device = next(_model.parameters()).device
    df_ml = _df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'dteday_parsed'], errors='ignore')
    features_scaled = _scaler_features.transform(df_ml.drop(columns=['cnt']))
    target_dummy = np.zeros((len(features_scaled), 1))
    data_scaled = np.hstack((features_scaled, target_dummy))
    
    X = []
    for i in range(len(data_scaled) - 24):
        X.append(data_scaled[i:(i + 24), :-1])
    X = np.array(X)
    
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    
    all_embeddings = []
    _model.eval()
    with torch.no_grad():
        for i in range(0, len(X_t), 1024):
            batch = X_t[i:i+1024]
            _, embs = _model(batch)
            all_embeddings.append(embs.cpu().numpy())
            
    return np.vstack(all_embeddings), X

# Inicialización de recursos
try:
    df_raw = load_data()
    model, scaler_features, scaler_target, device = load_ml_resources()
    all_embeddings, X_seq = precompute_embeddings(model, df_raw, scaler_features)
    resources_ok = True
except Exception as e:
    st.error(f"Error cargando los recursos del modelo: {e}")
    resources_ok = False

# ---------------------------------------------------------
# 4. Estructura de la Barra Lateral (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    # Encabezado Logo & Marca
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; margin-top: 10px;">
            <div style="padding: 10px; background-color: #0f172a; color: white; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bike"><circle cx="18.5" cy="17.5" r="3.5"/><circle cx="5.5" cy="17.5" r="3.5"/><circle cx="15" cy="5" r="1"/><path d="M12 17.5V14l-3-3 4-3 2 3h2"/></svg>
            </div>
            <div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <h1 style="font-size: 14px; font-weight: 700; color: #0f172a; margin: 0; line-height: 1;">BIKE CORE</h1>
                    <span style="font-size: 9px; background-color: #f1f5f9; color: #475569; font-weight: 700; padding: 2px 6px; border-radius: 4px; border: 1px solid #e2e8f0;">v1.2.0</span>
                </div>
                <p style="font-size: 10px; color: #94a3b8; margin: 4px 0 0 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">MLOps & Telemetry Dashboard</p>
            </div>
        </div>
        
        <div style="border-radius: 16px; overflow: hidden; margin-bottom: 20px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <img src="https://images.unsplash.com/photo-1485965120184-e220f721d03e?auto=format&fit=crop&w=400&q=80" style="width: 100%; display: block;" />
        </div>
    """, unsafe_allow_html=True)
    
    # Especificaciones Técnicas como tarjeta estilizada
    st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <h4 style="font-size: 11px; font-weight: 750; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 12px 0; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>
                Especificaciones Técnicas
            </h4>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; border-bottom: 1px solid #f8fafc; padding-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 500;">Modelo:</span>
                    <span style="color: #0f172a; font-weight: 600; font-family: monospace;">LSTM (2 capas)</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; border-bottom: 1px solid #f8fafc; padding-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 500;">Dimensión Latente:</span>
                    <span style="color: #0f172a; font-weight: 600; font-family: monospace;">64 Embeddings</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; border-bottom: 1px solid #f8fafc; padding-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 500;">Lookback Window:</span>
                    <span style="color: #0f172a; font-weight: 600; font-family: monospace;">24 horas</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; border-bottom: 1px solid #f8fafc; padding-bottom: 6px;">
                    <span style="color: #64748b; font-weight: 500;">Métrica AutoML:</span>
                    <span style="color: #0f172a; font-weight: 600; font-family: monospace;">MAE = 21.58</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px;">
                    <span style="color: #64748b; font-weight: 500;">Origen de Datos:</span>
                    <span style="color: #0f172a; font-weight: 600;">UCI Dataset</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Créditos del pie de página
    st.markdown("""
        <div style="padding: 16px; border-top: 1px solid #f1f5f9; background-color: #f8fafc; border-radius: 12px; margin-top: auto;">
            <p style="font-size: 9px; text-transform: uppercase; font-weight: 700; color: #94a3b8; margin: 0; letter-spacing: 0.05em;">Desarrollado Por</p>
            <p style="font-size: 11px; font-weight: 700; color: #334155; margin: 2px 0 0 0;">Grupo 5 - Computacion</p>
            <p style="font-size: 9px; font-weight: 500; color: #64748b; margin: 2px 0 0 0;">Facultad de Ingenieria de Datos</p>
        </div>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------
# 5. Cuerpo Principal y Pestañas
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🚲 Panel de Control de Movilidad Urbana</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predicción de Demanda de Bicicletas, Recuperación Semántica y Telemetría de Modelos</p>", unsafe_allow_html=True)

if resources_ok:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Inferencia Interactiva por Lotes",
        "🔍 Motor de Búsqueda de Escenarios Homólogos",
        "📈 Telemetría y Alertas de Data Drift",
        "🔮 Pronóstico de Demanda Futura"
    ])
    
    # ---------------------------------------------------------
    # TAB 1: Inferencia Interactiva por Lotes
    # ---------------------------------------------------------
    with tab1:
        st.header("Predicción por Lotes y Registro MLOps")
        st.write("Selecciona un rango de fechas del dataset histórico para realizar predicciones y evaluar el rendimiento del modelo en tiempo real.")
        
        # Filtros de fecha
        col1, col2 = st.columns(2)
        with col1:
            min_date = df_raw['dteday_parsed'].min().date()
            max_date = df_raw['dteday_parsed'].max().date()
            start_date = st.date_input("Fecha de Inicio", min_value=min_date, max_value=max_date, value=datetime.date(2012, 6, 1))
        with col2:
            end_date = st.date_input("Fecha Fin", min_value=min_date, max_value=max_date, value=datetime.date(2012, 6, 7))
            
        if start_date > end_date:
            st.error("Error: La fecha de inicio no puede ser posterior a la fecha fin.")
        else:
            # Filtrar datos de hour.csv para predicción
            df_filtered = df_raw[(df_raw['dteday_parsed'].dt.date >= start_date) & (df_raw['dteday_parsed'].dt.date <= end_date)].copy()
            
            if len(df_filtered) == 0:
                st.warning("No hay datos disponibles para el rango de fechas seleccionado.")
            else:
                st.write(f"Filas encontradas para evaluar: **{len(df_filtered)} registros**")
                
                # Ejecutar predicciones secuenciales
                with st.spinner("Ejecutando inferencia sobre el espacio temporal..."):
                    indices = df_filtered.index.tolist()
                    preds = []
                    actuals = []
                    valid_indices = []
                    
                    for idx in indices:
                        # Asegurarse de tener 24 horas previas para el lookback
                        if idx >= 24:
                            seq_data = df_raw.iloc[idx-24:idx]
                            df_ml_seq = seq_data.drop(columns=['instant', 'dteday', 'casual', 'registered', 'dteday_parsed'], errors='ignore')
                            
                            # Escalar
                            seq_scaled_features = scaler_features.transform(df_ml_seq.drop(columns=['cnt']))
                            seq_target_dummy = scaler_target.transform(seq_data[['cnt']])
                            seq_scaled = np.hstack((seq_scaled_features, seq_target_dummy))
                            
                            # Convertir a tensor
                            seq_tensor = torch.tensor(seq_scaled[:, :-1], dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output_scaled, _ = model(seq_tensor)
                                pred_val = scaler_target.inverse_transform(output_scaled.cpu().numpy())[0][0]
                                preds.append(max(0, float(pred_val)))
                                actuals.append(df_raw.iloc[idx]['cnt'])
                                valid_indices.append(idx)
                                
                if len(preds) > 0:
                    df_results = df_raw.iloc[valid_indices].copy()
                    df_results['Demanda_Predicha'] = np.round(preds).astype(int)
                    df_results['Demanda_Real'] = actuals
                    
                    # Calcular métricas del lote
                    errors = np.array(actuals) - np.array(preds)
                    mae_batch = np.mean(np.abs(errors))
                    rmse_batch = np.sqrt(np.mean(errors**2))
                    
                    # Mostrar métricas del lote en tarjetas
                    metrics_html = f"""
                    <div class="kpis-grid">
                        <div class="premium-card">
                            <div class="metric-icon-container" style="background-color: #e0e7ff; color: #4338ca;">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>
                            </div>
                            <div>
                                <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">MAE del Lote</p>
                                <div class="metric-value-indigo">{mae_batch:.1f}</div>
                                <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Bicicletas de error promedio</p>
                            </div>
                        </div>
                        <div class="premium-card">
                            <div class="metric-icon-container" style="background-color: #f5f3ff; color: #6d28d9;">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                            </div>
                            <div>
                                <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">RMSE del Lote</p>
                                <div class="metric-value-violet">{rmse_batch:.1f}</div>
                                <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Sensible a errores grandes</p>
                            </div>
                        </div>
                        <div class="premium-card">
                            <div class="metric-icon-container" style="background-color: #ecfdf5; color: #047857;">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>
                            </div>
                            <div>
                                <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Muestras Procesadas</p>
                                <div class="metric-value-emerald">{len(df_results)}</div>
                                <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Horas continuas procesadas</p>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # Gráficos del Lote
                    g_col1, g_col2 = st.columns([2, 1])
                    
                    with g_col1:
                        st.markdown("""
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; margin-top: 12px;">
                                <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">📈 Curva de Comparación Temporal</h3>
                                <span style="font-size: 9px; background-color: #f1f5f9; color: #64748b; font-weight: 700; padding: 2px 10px; border-radius: 9999px;">LSTM vs Real</span>
                            </div>
                        """, unsafe_allow_html=True)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_results['dteday'] + " " + df_results['hr'].astype(str) + ":00", 
                                                 y=df_results['Demanda_Real'], 
                                                 mode='lines+markers', 
                                                 name='Demanda Real (UCI)',
                                                 line=dict(color='#0f172a', width=2),
                                                 marker=dict(size=4)))
                        fig.add_trace(go.Scatter(x=df_results['dteday'] + " " + df_results['hr'].astype(str) + ":00", 
                                                 y=df_results['Demanda_Predicha'], 
                                                 mode='lines', 
                                                 name='Predicción LSTM',
                                                 line=dict(color='#6366f1', width=1.5, dash='dash')))
                        fig.update_layout(
                            xaxis_title="Fecha y Hora",
                            yaxis_title="Bicicletas Alquiladas",
                            hovermode="x unified",
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            font=dict(family='Outfit', size=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color='#64748b'))
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                    with g_col2:
                        st.markdown("""
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; margin-top: 12px;">
                                <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">📊 Distribución de Residuos</h3>
                                <span style="font-size: 9px; background-color: #f1f5f9; color: #64748b; font-weight: 700; padding: 2px 10px; border-radius: 9999px;">Frecuencia</span>
                            </div>
                        """, unsafe_allow_html=True)
                        fig_res = px.histogram(
                            x=errors,
                            nbins=15,
                            labels={'x': 'Error Absoluto (Bicicletas)'},
                            color_discrete_sequence=['#0f172a']
                        )
                        fig_res.update_layout(
                            yaxis_title="Frecuencia",
                            showlegend=False,
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            font=dict(family='Outfit', size=10)
                        )
                        st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': False})
                    
                    # Descargar y Guardar en Predictions Store
                    st.markdown("""
                        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; display: flex; flex-direction: column; gap: 12px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #64748b;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                                <h4 style="font-size: 12px; font-weight: 700; color: #0f172a; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">Auditoría e Ingesta MLOps</h4>
                            </div>
                            <p style="font-size: 11px; color: #64748b; margin: 0;">
                                Exporte el lote activo a CSV para análisis externo o regístrelo en el Predictions Store persistido en formato JSON.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    col_dl, col_log = st.columns(2)
                    with col_dl:
                        csv_data = df_results[['dteday', 'hr', 'Demanda_Real', 'Demanda_Predicha']].to_csv(index=False)
                        st.download_button(
                            label="📥 Exportar CSV",
                            data=csv_data,
                            file_name=f"predicciones_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
                    with col_log:
                        if st.button("➕ Registrar en Store"):
                            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            batch_num = random.randint(10000, 99999)
                            batch_id = f"batch_{batch_num}"
                            
                            pred_records = []
                            for idx, row in df_results.iterrows():
                                pred_records.append({
                                    "hora": int(row['hr']),
                                    "demanda_predicha": int(row['Demanda_Predicha'])
                                })
                                
                            log_content = {
                                "timestamp_ejecucion": timestamp_str,
                                "batch_id": batch_id,
                                "startDate": start_date.strftime("%Y-%m-%d"),
                                "endDate": end_date.strftime("%Y-%m-%d"),
                                "recordCount": len(df_results),
                                "avgMae": round(float(mae_batch), 1),
                                "avgRmse": round(float(rmse_batch), 1),
                                "predicciones": pred_records
                            }
                            
                            log_filename = f"log_pred_{batch_num}.json"
                            log_filepath = os.path.join(PRED_STORE_DIR, log_filename)
                            
                            os.makedirs(PRED_STORE_DIR, exist_ok=True)
                            with open(log_filepath, "w", encoding="utf-8") as f:
                                json.dump(log_content, f, indent=4)
                                
                            st.success(f"¡Lote guardado con éxito! Registrado como {batch_id}")
                            st.rerun()

                    # predictions_store loader & renderer
                    saved_batches = []
                    if os.path.exists(PRED_STORE_DIR):
                        for filename in sorted(os.listdir(PRED_STORE_DIR), reverse=True):
                            if filename.endswith(".json"):
                                filepath = os.path.join(PRED_STORE_DIR, filename)
                                try:
                                    with open(filepath, "r", encoding="utf-8") as f:
                                        log = json.load(f)
                                        saved_batches.append({
                                            "id": log.get("batch_id", filename.replace("log_pred_", "").replace(".json", "")),
                                            "timestamp": log.get("timestamp_ejecucion", "N/A"),
                                            "startDate": log.get("startDate", "N/A"),
                                            "endDate": log.get("endDate", "N/A"),
                                            "recordCount": log.get("recordCount", len(log.get("predicciones", []))),
                                            "avgMae": log.get("avgMae", "N/A"),
                                            "avgRmse": log.get("avgRmse", "N/A")
                                        })
                                except:
                                    pass

                    table_rows = ""
                    for b in saved_batches:
                        mae_str = f"{b['avgMae']}" if b['avgMae'] != "N/A" else "N/A"
                        rmse_str = f"{b['avgRmse']}" if b['avgRmse'] != "N/A" else "N/A"
                        table_rows += textwrap.dedent(f"""
                            <tr style="border-bottom: 1px solid #f1f5f9;">
                                <td style="padding: 10px 12px; font-family: monospace; font-weight: 700; color: #0f172a;">{b['id']}</td>
                                <td style="padding: 10px 12px; color: #64748b;">{b['timestamp']}</td>
                                <td style="padding: 10px 12px; color: #334155; font-weight: 500;">{b['startDate']} al {b['endDate']}</td>
                                <td style="padding: 10px 12px; text-align: center; font-weight: 600; font-family: monospace; color: #475569;">{b['recordCount']}</td>
                                <td style="padding: 10px 12px; text-align: center; font-weight: 700; font-family: monospace; color: #0f172a;">{mae_str}</td>
                                <td style="padding: 10px 12px; text-align: center; font-weight: 700; font-family: monospace; color: #6366f1;">{rmse_str}</td>
                            </tr>
                        """)

                    table_html = textwrap.dedent(f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 16px; overflow: hidden; background: white; margin-top: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                        <div style="padding: 16px; border-bottom: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0; display: flex; align-items: center; gap: 8px;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #64748b;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>
                                Predictions Store
                            </h3>
                            <span style="font-size: 10px; background-color: #f1f5f9; color: #475569; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">
                                {len(saved_batches)} Lotes Guardados
                            </span>
                        </div>
                        <div style="overflow-x: auto;">
                            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 11px;">
                                <thead>
                                    <tr style="background-color: #f8fafc; border-bottom: 1px solid #f1f5f9; color: #94a3b8; font-weight: 700; text-transform: uppercase; font-size: 10px; letter-spacing: 0.025em;">
                                        <th style="padding: 10px 12px;">ID Lote</th>
                                        <th style="padding: 10px 12px;">Fecha Ejecución</th>
                                        <th style="padding: 10px 12px;">Periodo Evaluado</th>
                                        <th style="padding: 10px 12px; text-align: center;">Horas</th>
                                        <th style="padding: 10px 12px; text-align: center;">MAE Promedio</th>
                                        <th style="padding: 10px 12px; text-align: center;">RMSE Promedio</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {table_rows if table_rows else '<tr><td colspan="6" style="padding: 24px; text-align: center; color: #94a3b8;">No hay lotes guardados aún</td></tr>'}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    """)
                    st.markdown(clean_html(table_html), unsafe_allow_html=True)
                    
                    if len(saved_batches) > 0:
                        with st.expander("🛠️ Administrar Predictions Store (Eliminar Lotes)"):
                            col_del_sel, col_del_btn = st.columns([3, 1])
                            with col_del_sel:
                                delete_target = st.selectbox(
                                    "Seleccione el lote que desea eliminar:",
                                    options=[b['id'] for b in saved_batches],
                                    key="delete_batch_sel"
                                )
                            with col_del_btn:
                                st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                                if st.button("🗑️ Eliminar Lote"):
                                    target_file = None
                                    for filename in os.listdir(PRED_STORE_DIR):
                                        if filename.endswith(".json"):
                                            filepath = os.path.join(PRED_STORE_DIR, filename)
                                            try:
                                                with open(filepath, "r", encoding="utf-8") as f:
                                                    log = json.load(f)
                                                    if log.get("batch_id") == delete_target or filename.replace("log_pred_", "").replace(".json", "") == delete_target:
                                                        target_file = filepath
                                                        break
                                            except:
                                                pass
                                    if target_file and os.path.exists(target_file):
                                        os.remove(target_file)
                                        st.success(f"Lote {delete_target} eliminado.")
                                        st.rerun()

                    st.markdown("""
                        <div style="display: flex; align-items: center; gap: 8px; margin-top: 24px; margin-bottom: 8px;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #64748b;"><path d="M12 20h.01"/><path d="M12 16h.01"/><path d="M12 12h.01"/><path d="M12 8h.01"/><path d="M12 4h.01"/><path d="M8 20h.01"/><path d="M8 16h.01"/><path d="M8 12h.01"/><path d="M8 8h.01"/><path d="M8 4h.01"/><path d="M16 20h.01"/><path d="M16 16h.01"/><path d="M16 12h.01"/><path d="M16 8h.01"/><path d="M16 4h.01"/><path d="M20 20h.01"/><path d="M20 16h.01"/><path d="M20 12h.01"/><path d="M20 8h.01"/><path d="M20 4h.01"/><path d="M4 20h.01"/><path d="M4 16h.01"/><path d="M4 12h.01"/><path d="M4 8h.01"/><path d="M4 4h.01"/></svg>
                            <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">📋 Vista Previa de los Datos Evaluados</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df_results[['dteday', 'hr', 'temp', 'hum', 'windspeed', 'Demanda_Real', 'Demanda_Predicha']].style.highlight_max(axis=0, subset=['Demanda_Real', 'Demanda_Predicha'], color='#ffebec'))
                else:
                    st.error("No se pudieron generar secuencias válidas. Por favor selecciona un rango de fechas con más registros históricos.")

    # ---------------------------------------------------------
    # TAB 2: Motor de Búsqueda de Escenarios Homólogos
    # ---------------------------------------------------------
    with tab2:
        st.header("Búsqueda Semántica en el Espacio Latente")
        
        # Explicación clara y premium de qué son los homólogos
        st.markdown("""
        <div style="background-color: #f0fdfa; border-left: 5px solid #0d9488; border-radius: 12px; padding: 16px; margin-bottom: 24px;">
            <p style="font-size: 13px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0; display: flex; align-items: center; gap: 6px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #0d9488;"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                ¿Qué es un Escenario Homólogo y cómo funciona esta tecnología?
            </p>
            <p style="font-size: 11px; color: #334155; margin: 0; line-height: 1.4;">
                Un <strong>escenario homólogo</strong> es una ventana temporal en el pasado que se comportó de manera casi idéntica a la fecha que estás consultando. En lugar de hacer una búsqueda simple por palabras o comparar números de forma aislada, utilizamos la <strong>red neuronal LSTM</strong>:
            </p>
            <ol style="font-size: 11px; color: #334155; margin: 6px 0 0 0; padding-left: 20px; line-height: 1.4;">
                <li>La LSTM procesa las 24 horas del día seleccionado (temperatura, viento, humedad, etc.) y las traduce en un <strong>código matemático compacto de 64 números</strong> (llamado <i>Embedding</i> o representación en el <i>Espacio Latente</i>).</li>
                <li>Este código captura la "esencia" y el patrón temporal de ese día.</li>
                <li>Calculamos la <strong>distancia Euclidiana</strong> entre el código del día de consulta y los códigos de todo el historial para hallar cuáles son los días matemáticamente más similares (los homólogos).</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Selección del escenario de consulta (Query Scenario)
        st.write("### 🎛️ Definir Escenario de Consulta")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            q_date = st.date_input("Fecha de Consulta", min_value=min_date, max_value=max_date, value=datetime.date(2012, 10, 15), key="q_date")
        with q_col2:
            q_hour = st.slider("Hora de Consulta", 0, 23, 14, key="q_hour")
            
        # Buscar el índice correspondiente al escenario en df_raw
        query_rows = df_raw[(df_raw['dteday_parsed'].dt.date == q_date) & (df_raw['hr'] == q_hour)]
        
        if len(query_rows) == 0:
            st.warning("No se encontró ningún registro para la fecha y hora de consulta seleccionadas en el historial.")
        else:
            q_idx = query_rows.index[0]
            
            if q_idx < 24:
                st.error("El registro seleccionado se encuentra en las primeras 24 horas del dataset y no tiene suficiente historial para construir una secuencia de entrada.")
            else:
                st.success(f"Escenario de Consulta localizado (Índice en BD: {q_idx})")
                
                # Extraer embedding del query
                q_emb_idx = q_idx - 24
                q_emb = all_embeddings[q_emb_idx]
                
                # Calcular distancias Euclidianas frente a todos los demás embeddings
                query_date_str = df_raw.iloc[q_idx]['dteday']
                
                distances = []
                for i in range(len(all_embeddings)):
                    real_idx = i + 24
                    if df_raw.iloc[real_idx]['dteday'] == query_date_str:
                        distances.append(float('inf'))
                    else:
                        dist = np.linalg.norm(all_embeddings[i] - q_emb)
                        distances.append(dist)
                        
                distances = np.array(distances)
                
                # Obtener los 3 escenarios más homólogos (menor distancia)
                top_k_indices = np.argsort(distances)[:3]
                
                # Mostrar el escenario consultado
                st.write("### 🎯 Contraste del Escenario de Consulta vs Vecinos Homólogos")
                
                cols = st.columns(4)
                
                # Tarjeta del Query
                with cols[0]:
                    row_q = df_raw.iloc[q_idx]
                    card_q = f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #0f172a; border-radius: 16px; padding: 16px; min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                                <span style="font-size: 13px;">🎯</span>
                                <h4 style="font-size: 11px; font-weight: 750; color: #0f172a; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">Escenario Consulta</h4>
                            </div>
                            <p style="margin: 3px 0; font-size: 11px; color: #475569;"><strong style="color: #0f172a;">Fecha:</strong> {row_q['dteday']}</p>
                            <p style="margin: 3px 0; font-size: 11px; color: #475569;"><strong style="color: #0f172a;">Hora:</strong> {row_q['hr']}:00 h</p>
                            <p style="margin: 3px 0; font-size: 11px; color: #475569;"><strong style="color: #0f172a;">Temp:</strong> {row_q['temp'] * 41:.1f} °C</p>
                            <p style="margin: 3px 0; font-size: 11px; color: #475569;"><strong style="color: #0f172a;">Hum:</strong> {row_q['hum'] * 100:.1f} %</p>
                            <p style="margin: 3px 0; font-size: 11px; color: #475569;"><strong style="color: #0f172a;">Viento:</strong> {row_q['windspeed'] * 67:.1f} km/h</p>
                        </div>
                        <div style="margin-top: 10px; border-top: 1px solid #f1f5f9; padding-top: 8px;">
                            <span style="font-size: 9px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Demanda Real</span>
                            <div style="font-size: 18px; font-weight: 800; color: #0f172a; line-height: 1;">{row_q['cnt']} <span style="font-size: 10px; font-weight: 500; color: #64748b;">bicis</span></div>
                        </div>
                    </div>
                    """
                    st.markdown(card_q, unsafe_allow_html=True)
                    
                # Mostrar los 3 vecinos homólogos
                for rank, j in enumerate(top_k_indices):
                    real_match_idx = j + 24
                    row_m = df_raw.iloc[real_match_idx]
                    dist_val = distances[j]
                    
                    with cols[rank + 1]:
                        card_m = f"""
                        <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-left: 5px solid #f59e0b; border-radius: 16px; padding: 16px; min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
                            <div>
                                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                                    <span style="font-size: 13px;">🏆</span>
                                    <h4 style="font-size: 11px; font-weight: 750; color: #b45309; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">Homólogo #{rank+1}</h4>
                                </div>
                                <p style="margin: 3px 0; font-size: 11px; color: #78350f;"><strong style="color: #b45309;">Fecha:</strong> {row_m['dteday']}</p>
                                <p style="margin: 3px 0; font-size: 11px; color: #78350f;"><strong style="color: #b45309;">Hora:</strong> {row_m['hr']}:00 h</p>
                                <p style="margin: 3px 0; font-size: 11px; color: #78350f;"><strong style="color: #b45309;">Temp:</strong> {row_m['temp'] * 41:.1f} °C</p>
                                <p style="margin: 3px 0; font-size: 11px; color: #78350f;"><strong style="color: #b45309;">Hum:</strong> {row_m['hum'] * 100:.1f} %</p>
                                <p style="margin: 3px 0; font-size: 10px; color: #78350f; font-family: monospace; background: rgba(254, 243, 199, 0.6); padding: 2px 6px; border-radius: 6px; display: inline-block;"><strong style="color: #b45309;">Distancia:</strong> {dist_val:.4f}</p>
                            </div>
                            <div style="margin-top: 10px; border-top: 1px solid #fef3c7; padding-top: 8px;">
                                <span style="font-size: 9px; color: #b45309; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Demanda Histórica</span>
                                <div style="font-size: 18px; font-weight: 800; color: #b45309; line-height: 1;">{row_m['cnt']} <span style="font-size: 10px; font-weight: 500; color: #b45309;">bicis</span></div>
                            </div>
                        </div>
                        """
                        st.markdown(card_m, unsafe_allow_html=True)
                
                # Gráficos comparativos de las 24 horas del perfil
                st.write("### 📈 Visualización del Perfil de Inercia (Ventana de 24 horas)")
                
                compare_data = []
                
                q_window = df_raw.iloc[q_idx-24:q_idx].copy()
                q_window['Escenario'] = f"Consulta ({query_date_str})"
                q_window['Paso_Tiempo'] = range(24)
                compare_data.append(q_window[['Paso_Tiempo', 'temp', 'hum', 'cnt', 'Escenario']])
                
                for rank, j in enumerate(top_k_indices):
                    real_match_idx = j + 24
                    m_date_str = df_raw.iloc[real_match_idx]['dteday']
                    m_window = df_raw.iloc[real_match_idx-24:real_match_idx].copy()
                    m_window['Escenario'] = f"Homólogo #{rank+1} ({m_date_str})"
                    m_window['Paso_Tiempo'] = range(24)
                    compare_data.append(m_window[['Paso_Tiempo', 'temp', 'hum', 'cnt', 'Escenario']])
                    
                df_compare_plot = pd.concat(compare_data)
                
                pcol1, pcol2 = st.columns(2)
                
                with pcol1:
                    st.markdown("""
                        <h4 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Perfil de Demanda (Bicicletas)</h4>
                    """, unsafe_allow_html=True)
                    fig_compare_dem = px.line(
                        df_compare_plot, 
                        x='Paso_Tiempo', 
                        y='cnt', 
                        color='Escenario',
                        labels={'Paso_Tiempo': 'Horas de la Ventana (0-23)', 'cnt': 'Bicicletas Alquiladas'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_compare_dem.update_layout(
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                        yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                        font=dict(family='Outfit', size=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8, color='#64748b'))
                    )
                    st.plotly_chart(fig_compare_dem, use_container_width=True, config={'displayModeBar': False})
                    
                with pcol2:
                    st.markdown("""
                        <h4 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Perfil de Temperatura (°C)</h4>
                    """, unsafe_allow_html=True)
                    df_compare_plot['temp_celsius'] = df_compare_plot['temp'] * 41.0
                    fig_compare_temp = px.line(
                        df_compare_plot, 
                        x='Paso_Tiempo', 
                        y='temp_celsius', 
                        color='Escenario',
                        labels={'Paso_Tiempo': 'Horas de la Ventana (0-23)', 'temp_celsius': 'Temperatura (°C)'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_compare_temp.update_layout(
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                        yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                        font=dict(family='Outfit', size=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8, color='#64748b'))
                    )
                    st.plotly_chart(fig_compare_temp, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------------------------------------
    # TAB 3: Tablero de Control de Telemetría y Alertas de Data Drift
    # ---------------------------------------------------------
    with tab3:
        st.header("Rendimiento del Modelo y Telemetría de Desviación")
        st.write("Este panel monitorea las métricas diarias del modelo en producción. El sistema dispara alertas de Data Drift cuando la media móvil del MAE (calculada sobre los últimos días evaluados) supera el umbral crítico de **35.0**.")
        
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = []
            
        if len(metrics_data) == 0:
            st.warning("No hay registros en el archivo de monitoreo diario.")
        else:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics['fecha'] = pd.to_datetime(df_metrics['fecha'])
            df_metrics = df_metrics.sort_values(by='fecha')
            
            # Formatear todas las fechas disponibles
            all_dates = df_metrics['fecha'].dt.strftime("%Y-%m-%d").tolist()
            
            # Asegurar que st.session_state tenga una fecha activa válida y por defecto sea la última cronológica
            if "active_telemetry_date" not in st.session_state:
                st.session_state["active_telemetry_date"] = all_dates[-1]
            elif st.session_state["active_telemetry_date"] not in all_dates:
                st.session_state["active_telemetry_date"] = all_dates[-1]
                
            # Selector interactivo de fecha para visualización en el dashboard
            sel_telemetry_col1, sel_telemetry_col2 = st.columns([1, 1])
            with sel_telemetry_col1:
                active_date_str = st.selectbox(
                    "📅 Seleccione Fecha de Evaluación para el Dashboard:",
                    options=all_dates,
                    index=all_dates.index(st.session_state["active_telemetry_date"]),
                    key="telemetry_date_selector_widget"
                )
                st.session_state["active_telemetry_date"] = active_date_str
            
            # Obtener la fila correspondiente a la fecha seleccionada
            latest_row = df_metrics[df_metrics['fecha'].dt.strftime("%Y-%m-%d") == active_date_str].iloc[0]
            latest_mae_movil = float(latest_row['mae_movil'])
            latest_mae = float(latest_row['mae'])
            latest_rmse = float(latest_row['rmse'])
            
            # Alerta visual interactiva
            if latest_mae_movil > 35.0:
                st.markdown(f"""
                    <style>
                    @keyframes pulse-red {{
                        0%, 100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
                        50% {{ box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
                    }}
                    .pulsing-drift-alert {{
                        background-color: #fef2f2; 
                        border: 1px solid #fecaca; 
                        border-radius: 16px; 
                        padding: 20px; 
                        display: flex; 
                        flex-direction: column; 
                        gap: 16px; 
                        margin-bottom: 24px;
                        animation: pulse-red 2s infinite;
                    }}
                    @media (min-width: 768px) {{
                        .pulsing-drift-alert {{
                            flex-direction: row;
                            align-items: center;
                            justify-content: space-between;
                        }}
                    }}
                    </style>
                    <div class="pulsing-drift-alert">
                        <div style="display: flex; align-items: flex-start; gap: 16px;">
                            <div style="padding: 10px; background-color: #fee2e2; border-radius: 12px; border: 1px solid #fecaca; color: #dc2626; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            </div>
                            <div>
                                <h3 style="font-size: 14px; font-weight: 700; color: #7f1d1d; margin: 0;">🚨 ALERTA OPERACIONAL: Se detectó Data Drift</h3>
                                <p style="font-size: 11px; color: #991b1b; margin: 4px 0 0 0; font-weight: 500; line-height: 1.4;">
                                    La media móvil del error absoluto (MAE 7d) ha alcanzado un valor acumulado de <strong style="font-family: monospace;">{latest_mae_movil:.2f} MAE</strong>, superando el umbral de tolerancia operacional de <strong>35.0</strong>.
                                </p>
                                <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 11px; color: #991b1b; font-weight: 500; line-height: 1.4;">
                                    <li><strong>Degradación:</strong> Desvío en la asertividad predictiva bajo condiciones estacionales atípicas en demanda.</li>
                                    <li><strong>Mitigación sugerida:</strong> Iniciar pipeline automático de ajuste de hiperparámetros y reentrenamiento.</li>
                                </ul>
                            </div>
                        </div>
                        <div style="font-size: 10px; background-color: #fee2e2; color: #b91c1c; padding: 6px 12px; border-radius: 8px; border: 1px solid #fecaca; font-family: monospace; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; align-self: flex-start; margin-top: 8px;">
                            Estado: Drift Activo
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 16px; padding: 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px;">
                        <div style="display: flex; align-items: flex-start; gap: 16px;">
                            <div style="padding: 10px; background-color: #dcfce7; border-radius: 12px; border: 1px solid #bbf7d0; color: #15803d; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            </div>
                            <div>
                                <h3 style="font-size: 14px; font-weight: 700; color: #14532d; margin: 0;">🟢 Rendimiento Nominal: Estable</h3>
                                <p style="font-size: 11px; color: #166534; margin: 4px 0 0 0; font-weight: 500;">
                                    La media móvil del error ({latest_mae_movil:.2f} MAE) se encuentra en rangos tolerables. El modelo opera óptimamente.
                                </p>
                            </div>
                        </div>
                        <div style="font-size: 10px; background-color: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 8px; border: 1px solid #bbf7d0; font-family: monospace; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;">
                            Estado: Nominal
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Métricas superiores y velocímetro de Drift
            stat_col, gauge_col = st.columns([1, 1])
            
            with stat_col:
                st.markdown("""
                    <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px;">📊 Métricas del Último Día de Evaluación</h3>
                """, unsafe_allow_html=True)
                kpi_telemetry_html = f"""
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <div class="premium-card">
                        <div class="metric-icon-container" style="background-color: #e0e7ff; color: #4338ca;">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                        </div>
                        <div>
                            <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Último MAE Diario</p>
                            <div class="metric-value-indigo">{latest_mae:.2f}</div>
                            <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Día registrado: {latest_row['fecha'].strftime('%Y-%m-%d')}</p>
                        </div>
                    </div>
                    <div class="premium-card">
                        <div class="metric-icon-container" style="background-color: #f5f3ff; color: #6d28d9;">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/></svg>
                        </div>
                        <div>
                            <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Último RMSE Diario</p>
                            <div class="metric-value-violet">{latest_rmse:.2f}</div>
                            <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Día registrado: {latest_row['fecha'].strftime('%Y-%m-%d')}</p>
                        </div>
                    </div>
                </div>
                """
                st.markdown(kpi_telemetry_html, unsafe_allow_html=True)
                
            with gauge_col:
                # Indicador de Velocímetro Premium para Data Drift
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = latest_mae_movil,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Velocímetro de Deriva (MAE Móvil 7d)", 'font': {'size': 12, 'family': 'Outfit', 'color': '#0f172a', 'weight': 'bold'}},
                    delta = {'reference': 35.0, 'increasing': {'color': "#dc2626"}, 'decreasing': {'color': "#16803d"}},
                    gauge = {
                        'axis': {'range': [0, 60], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'family': 'Outfit', 'size': 9, 'color': '#94a3b8'}},
                        'bar': {'color': "#0f172a"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 25], 'color': '#dcfce7'},
                            {'range': [25, 35], 'color': '#fef3c7'},
                            {'range': [35, 60], 'color': '#fee2e2'}
                        ],
                        'threshold': {
                            'line': {'color': "#dc2626", 'width': 3},
                            'thickness': 0.75,
                            'value': 35.0
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=240, 
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Outfit', size=11)
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                
            # Gráfico de control
            st.markdown("""
                <div style="margin-top: 16px; margin-bottom: 8px;">
                    <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">📉 Gráfico de Monitoreo de Deriva Temporal</h3>
                </div>
            """, unsafe_allow_html=True)
            fig_telemetry = go.Figure()
            fig_telemetry.add_trace(go.Scatter(x=df_metrics['fecha'], y=df_metrics['mae'],
                                                mode='lines+markers', name='MAE Diario',
                                                line=dict(color='#0f172a', width=1.5),
                                                marker=dict(size=4)))
            fig_telemetry.add_trace(go.Scatter(x=df_metrics['fecha'], y=df_metrics['mae_movil'],
                                                mode='lines', name='MAE Móvil (7d)',
                                                line=dict(color='#6366f1', width=2)))
            fig_telemetry.add_trace(go.Scatter(x=df_metrics['fecha'], y=[35.0]*len(df_metrics),
                                                mode='lines', name='Límite de Alerta (35.0)',
                                                line=dict(color='#ef4444', width=1.5, dash='dash')))
            fig_telemetry.update_layout(
                xaxis_title="Fecha de Monitoreo",
                yaxis_title="Bicicletas (Valor de Métrica)",
                hovermode="x unified",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                font=dict(family='Outfit', size=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8, color='#64748b'))
            )
            st.plotly_chart(fig_telemetry, use_container_width=True, config={'displayModeBar': False})
            
            # Formulario interactivo para simular evaluaciones diarias (Sliders Premium)
            st.markdown("---")
            res_col1, res_col2 = st.columns([3, 1])
            with res_col1:
                st.write("#### 🧪 Simulador de Monitoreo Diario")
            with res_col2:
                if st.button("🗑️ Restablecer Telemetría", help="Limpia todos los registros del historial de monitoreo diario."):
                    initial_defaults = [
                        {"fecha": "2026-06-08", "mae": 15.2, "rmse": 22.1, "mae_movil": 15.20, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-09", "mae": 18.5, "rmse": 25.3, "mae_movil": 16.85, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-10", "mae": 14.8, "rmse": 20.9, "mae_movil": 16.17, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-11", "mae": 16.1, "rmse": 23.4, "mae_movil": 16.15, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-12", "mae": 19.3, "rmse": 27.8, "mae_movil": 16.78, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-13", "mae": 15.9, "rmse": 21.6, "mae_movil": 16.63, "registros_evaluados": 24, "drift_detected": False},
                        {"fecha": "2026-06-14", "mae": 17.2, "rmse": 24.1, "mae_movil": 16.71, "registros_evaluados": 24, "drift_detected": False}
                    ]
                    with open(METRICS_PATH, "w", encoding="utf-8") as f:
                        json.dump(initial_defaults, f, indent=4)
                    st.session_state["active_telemetry_date"] = "2026-06-14"
                    st.success("Historial de telemetría restablecido a valores nominales.")
                    st.rerun()

            with st.expander("🧪 **Simular Nueva Evaluación Diaria (Probar Alerta de Data Drift)**", expanded=False):
                st.markdown("""
                Usa este simulador interactivo para registrar nuevos días de monitoreo y evaluar la estabilidad del modelo.
                * **Tip:** Desliza el **MAE del día** a un valor alto (ej. `> 45.0`) para ver cómo la media móvil cruza el umbral crítico de **35.0** y activa la alerta de Data Drift.
                """)
                
                with st.form("simular_evaluacion"):
                    sim_col1, sim_col2, sim_col3 = st.columns(3)
                    with sim_col1:
                        new_date = st.date_input("Fecha de Evaluación", value=datetime.date.today(), help="Elige la fecha para registrar la métrica simulada.")
                    with sim_col2:
                        new_mae = st.slider("MAE del día", min_value=5.0, max_value=120.0, value=25.0, step=0.5, help="Error absoluto medio diario.")
                    with sim_col3:
                        new_rmse = st.slider("RMSE del día", min_value=5.0, max_value=150.0, value=35.0, step=0.5, help="Error cuadrático medio diario.")
                        
                    submitted = st.form_submit_button("⚡ Registrar Métricas y Actualizar Telemetría")
                    
                    if submitted:
                        new_date_str = new_date.strftime("%Y-%m-%d")
                        
                        # Remove existing entry for the same date if it exists (overwrite behavior)
                        metrics_data = [m for m in metrics_data if m["fecha"] != new_date_str]
                        
                        # Add new entry temporarily
                        temp_record = {
                            "fecha": new_date_str,
                            "mae": float(new_mae),
                            "rmse": float(new_rmse),
                            "mae_movil": 0.0,  # will recalculate
                            "registros_evaluados": 24,
                            "drift_detected": False  # will recalculate
                        }
                        metrics_data.append(temp_record)
                        
                        # Sort chronologically by date
                        metrics_data = sorted(metrics_data, key=lambda x: x["fecha"])
                        
                        # Recalculate mae_movil and drift_detected for all records chronologically
                        for i in range(len(metrics_data)):
                            # Take the last 7 records up to index i
                            start_window_idx = max(0, i - 6)
                            window_records = metrics_data[start_window_idx : i + 1]
                            maes_in_window = [r["mae"] for r in window_records]
                            
                            running_mae_movil = float(np.mean(maes_in_window))
                            metrics_data[i]["mae_movil"] = round(running_mae_movil, 2)
                            metrics_data[i]["drift_detected"] = running_mae_movil > 35.0
                            
                        # Save back to JSON
                        with open(METRICS_PATH, "w", encoding="utf-8") as f:
                            json.dump(metrics_data, f, indent=4)
                            
                        # Set active date in session state to the newly registered date
                        st.session_state["active_telemetry_date"] = new_date_str
                        st.success(f"¡Registro del {new_date_str} guardado con éxito! Recargando el panel...")
                        st.rerun()
                        
            # Tabla de registros históricos
            st.write("### Historial Completo de Telemetría Registrada")
            st.dataframe(df_metrics[['fecha', 'mae', 'rmse', 'mae_movil', 'drift_detected']].style.format({
                'fecha': lambda t: t.strftime('%Y-%m-%d'),
                'mae': '{:.2f}',
                'rmse': '{:.2f}',
                'mae_movil': '{:.2f}'
            }).highlight_max(subset=['mae_movil'], color='#ffebec'))

    # ---------------------------------------------------------
    # TAB 4: Pronóstico de Demanda Futura (Forecasting)
    # ---------------------------------------------------------
    with tab4:
        st.header("🔮 Pronóstico y Simulación de Demanda Futura")
        st.write("Configura un punto de partida histórico ('Hoy') y simula las condiciones del clima de los siguientes días para proyectar la demanda de alquiler de bicicletas.")
        
        # 1. Configuración
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            f_date = st.date_input("Fecha de Partida (Hoy)", min_value=min_date, max_value=max_date - datetime.timedelta(days=3), value=datetime.date(2012, 12, 25), key="f_date")
        with fcol2:
            f_hour = st.slider("Hora de Partida", 0, 23, 12, key="f_hour")
            
        # Buscar el índice base en df_raw
        base_rows = df_raw[(df_raw['dteday_parsed'].dt.date == f_date) & (df_raw['hr'] == f_hour)]
        
        if len(base_rows) == 0:
            st.warning("No se encontró ningún registro para la fecha y hora seleccionadas.")
        else:
            base_idx = base_rows.index[0]
            
            if base_idx < 24:
                st.error("Por favor selecciona una fecha posterior para tener suficiente historial de lookback (mínimo 24 horas).")
            else:
                # Controles del Simulador Climático (What-If)
                st.markdown("### 🎛️ Simulador de Escenario Climático (What-If)")
                scol1, scol2, scol3 = st.columns(3)
                with scol1:
                    temp_delta = st.slider("Variación de Temperatura (°C)", -15.0, 15.0, 0.0, step=0.5, help="Suma o resta grados a la temperatura histórica de los próximos días.")
                with scol2:
                    hum_delta = st.slider("Variación de Humedad (%)", -30.0, 30.0, 0.0, step=1.0, help="Ajusta el porcentaje de humedad del aire.")
                with scol3:
                    weather_override = st.selectbox(
                        "Alterar Estado del Clima",
                        options=["Mantener Clima Histórico", "Despejado / Soleado", "Nublado / Neblina", "Lluvia Ligera / Nieve", "Tormenta Fuerte / Granizo"],
                        index=0
                    )
                
                horizon_hours = st.selectbox("Horizonte de Proyección", options=[24, 48, 72], index=0, format_func=lambda x: f"{x} horas ({x//24} días)")
                
                # Ejecutar Pronóstico
                if st.button("🔮 Generar Pronóstico Proyectado"):
                    with st.spinner("Computando inferencia secuencial LSTM..."):
                        forecast_records = []
                        
                        # Mapear weathersit
                        weather_map = {
                            "Despejado / Soleado": 1,
                            "Nublado / Neblina": 2,
                            "Lluvia Ligera / Nieve": 3,
                            "Tormenta Fuerte / Granizo": 4
                        }
                        
                        # Generar el pronóstico paso a paso
                        for k in range(horizon_hours):
                            target_idx = base_idx + k
                            
                            # Si superamos el tamaño de los datos históricos, rompemos
                            if target_idx >= len(df_raw):
                                break
                                
                            seq_data = df_raw.iloc[target_idx-24:target_idx].copy()
                            
                            # Modificar las horas que correspondan a la proyección del futuro (es decir, índice >= base_idx)
                            for idx_in_seq in range(24):
                                actual_global_idx = target_idx - 24 + idx_in_seq
                                if actual_global_idx >= base_idx:
                                    # Modificar temperatura
                                    original_temp_c = seq_data.iloc[idx_in_seq]['temp'] * 41.0
                                    new_temp_c = max(-8.0, min(39.0, original_temp_c + temp_delta))
                                    seq_data.iloc[idx_in_seq, seq_data.columns.get_loc('temp')] = new_temp_c / 41.0
                                    
                                    # Modificar atemp (feeling temperature)
                                    original_atemp_c = seq_data.iloc[idx_in_seq]['atemp'] * 50.0
                                    new_atemp_c = max(-16.0, min(50.0, original_atemp_c + temp_delta))
                                    seq_data.iloc[idx_in_seq, seq_data.columns.get_loc('atemp')] = new_atemp_c / 50.0
                                    
                                    # Modificar humedad
                                    original_hum = seq_data.iloc[idx_in_seq]['hum'] * 100.0
                                    new_hum = max(0.0, min(100.0, original_hum + hum_delta))
                                    seq_data.iloc[idx_in_seq, seq_data.columns.get_loc('hum')] = new_hum / 100.0
                                    
                                    # Modificar estado del clima
                                    if weather_override != "Mantener Clima Histórico":
                                        seq_data.iloc[idx_in_seq, seq_data.columns.get_loc('weathersit')] = weather_map[weather_override]
                            
                            # Preparar para el modelo LSTM
                            df_ml_seq = seq_data.drop(columns=['instant', 'dteday', 'casual', 'registered', 'dteday_parsed'], errors='ignore')
                            seq_scaled_features = scaler_features.transform(df_ml_seq.drop(columns=['cnt']))
                            seq_target_dummy = scaler_target.transform(seq_data[['cnt']])
                            seq_scaled = np.hstack((seq_scaled_features, seq_target_dummy))
                            
                            # Crear tensor (24 horas lookback, 12 features)
                            seq_tensor = torch.tensor(seq_scaled[:, :-1], dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output_scaled, _ = model(seq_tensor)
                                pred_val = scaler_target.inverse_transform(output_scaled.cpu().numpy())[0][0]
                                pred_cnt = max(0, int(np.round(pred_val)))
                                
                            row_target = df_raw.iloc[target_idx]
                            
                            # Obtener los datos climatológicos finales aplicados para esta hora proyectada
                            sim_temp_c = (seq_data.iloc[-1]['temp'] * 41.0) if (target_idx >= base_idx) else (row_target['temp'] * 41.0)
                            sim_hum_pct = (seq_data.iloc[-1]['hum'] * 100.0) if (target_idx >= base_idx) else (row_target['hum'] * 100.0)
                            
                            weathersit_val = int(seq_data.iloc[-1]['weathersit'])
                            weathersit_labels = {1: "Soleado", 2: "Nublado", 3: "Llovizna/Nieve", 4: "Tormenta"}
                            sim_weather = weathersit_labels.get(weathersit_val, "Nominal")
                            
                            forecast_records.append({
                                "Fecha/Hora": f"{row_target['dteday']} {int(row_target['hr'])}:00 h",
                                "Demanda_Real": int(row_target['cnt']),
                                "Demanda_Proyectada": pred_cnt,
                                "Temp_Simulada": sim_temp_c,
                                "Hum_Simulada": sim_hum_pct,
                                "Clima_Simulado": sim_weather
                            })
                            
                        # Construir DataFrame de resultados del pronóstico
                        df_fc = pd.DataFrame(forecast_records)
                        
                        # Métricas resumidas
                        total_real = df_fc['Demanda_Real'].sum()
                        total_proj = df_fc['Demanda_Proyectada'].sum()
                        avg_temp = df_fc['Temp_Simulada'].mean()
                        peak_hour = df_fc.loc[df_fc['Demanda_Proyectada'].idxmax()]['Fecha/Hora']
                        peak_val = df_fc['Demanda_Proyectada'].max()
                        
                        # Mostrar KPI Cards en HTML
                        kpi_fc_html = f"""
                        <div class="kpis-grid" style="margin-top: 16px;">
                            <div class="premium-card">
                                <div class="metric-icon-container" style="background-color: #e0e7ff; color: #4338ca;">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                                </div>
                                <div>
                                    <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Demanda Proyectada Total</p>
                                    <div class="metric-value-indigo">{total_proj:,}</div>
                                    <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Histórica real en este periodo: {total_real:,}</p>
                                </div>
                            </div>
                            <div class="premium-card">
                                <div class="metric-icon-container" style="background-color: #f5f3ff; color: #6d28d9;">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                </div>
                                <div>
                                    <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Pico de Demanda Proyectada</p>
                                    <div class="metric-value-violet">{peak_val} <span style="font-size: 14px; color: #64748b; font-weight: 500;">alquileres</span></div>
                                    <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Hora: {peak_hour}</p>
                                </div>
                            </div>
                            <div class="premium-card">
                                <div class="metric-icon-container" style="background-color: #ffe4e6; color: #e11d48;">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>
                                </div>
                                <div>
                                    <p style="font-size: 10px; font-weight: 750; color: #94a3b8; text-transform: uppercase; tracking-wider; margin: 0; letter-spacing: 0.05em;">Temperatura Promedio Simulada</p>
                                    <div class="metric-value-rose">{avg_temp:.1f} °C</div>
                                    <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; font-weight: 600;">Ajuste aplicado: {temp_delta:+.1f} °C</p>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(clean_html(kpi_fc_html), unsafe_allow_html=True)
                        
                        # Gráfico comparativo de curvas
                        st.markdown("""
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; margin-top: 24px;">
                                <h3 style="font-size: 12px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">📈 Comparación de Demanda: Pronóstico Simulada vs Histórica Real</h3>
                                <span style="font-size: 9px; background-color: #f1f5f9; color: #64748b; font-weight: 700; padding: 2px 10px; border-radius: 9999px;">Visualización de Escenario Proyectado</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(
                            x=df_fc['Fecha/Hora'],
                            y=df_fc['Demanda_Real'],
                            mode='lines+markers',
                            name='Demanda Histórica Real',
                            line=dict(color='#94a3b8', width=1.5),
                            marker=dict(size=4)
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=df_fc['Fecha/Hora'],
                            y=df_fc['Demanda_Proyectada'],
                            mode='lines+markers',
                            name='Pronóstico Proyectado (Simulado)',
                            line=dict(color='#6366f1', width=2.5),
                            marker=dict(size=5)
                        ))
                        fig_fc.update_layout(
                            xaxis_title="Periodo Futuro Proyectado",
                            yaxis_title="Cantidad de Alquileres",
                            hovermode="x unified",
                            height=380,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            yaxis=dict(gridcolor='#f1f5f9', showgrid=True, tickfont=dict(family='Outfit', size=9, color='#94a3b8'), title_font=dict(color='#64748b', size=10)),
                            font=dict(family='Outfit', size=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color='#64748b'))
                        )
                        st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})
                        
                        # Tabla de desglose de proyección
                        st.write("### 📋 Desglose del Escenario Proyectado por Horas")
                        st.dataframe(df_fc.style.format({
                            'Temp_Simulada': '{:.1f} °C',
                            'Hum_Simulada': '{:.1f} %',
                            'Demanda_Real': '{:,}',
                            'Demanda_Proyectada': '{:,}'
                        }).highlight_max(subset=['Demanda_Proyectada'], color='#e0e7ff'))

else:
    st.error("No se pudo iniciar la aplicación debido a que los recursos del modelo (pesos de la red LSTM o scalers) no se encuentran en la carpeta `/src/models/`. Por favor asegúrate de ejecutar el entrenamiento antes de correr la app.")
