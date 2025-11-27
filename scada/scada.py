import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
import os

# ---------------------------------
# CONFIG STREAMLIT
# ---------------------------------
st.set_page_config(
    page_title="SCADA - Caldera ENESA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------
# AUTO-REFRESH CADA 10 SEGUNDOS
# ---------------------------------
st_autorefresh(interval=10_000, key="scada_refresh")

# ---------------------------------
# ESTILOS
# ---------------------------------
st.markdown("""
    <style>
        body { background-color: #0e1117; color: #e1e1e1; }
        .card {
            padding: 20px;
            background-color: #161b22;
            border-radius: 12px;
            box-shadow: 1px 1px 8px rgba(255,255,255,0.08);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# TÍTULO
# ---------------------------------
st.markdown("<h1>SCADA – Caldera ENESA (Datos CTGAN)</h1>", unsafe_allow_html=True)

# ---------------------------------
# CARGAR DATOS (CON RUTA SEGURA)
# ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))             # Carpeta /scada
DATA_FILE = os.path.join(BASE, "..", "simulador", "stream_data.csv")

if not os.path.exists(DATA_FILE):
    st.warning(f"Esperando datos… No se encuentra el archivo:\n\n`{DATA_FILE}`")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Error al leer el archivo de datos:\n\n{e}")
    st.stop()

if df.empty:
    st.info("El archivo existe pero aún no contiene datos.")
    st.stop()

# ---------------------------------
# GRÁFICOS
# ---------------------------------

# Temperatura
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Temperatura de Gases (°C)")
st.plotly_chart(
    px.line(df, y="temperatura_gases_salida_c"),
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Oxígeno & Humedad
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Oxígeno (%)")
    st.plotly_chart(
        px.line(df, y="oxigeno_porcentaje_base_seca"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Humedad (%)")
    st.plotly_chart(
        px.line(df, y="humedad_porcentaje"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Material particulado
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Material Particulado (mg/m³)")
st.plotly_chart(
    px.line(df, y="concentracion_mp_mg_m3"),
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Flujos
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Flujo Húmedo (m³/min)")
    st.plotly_chart(
        px.line(df, y="flujo_gases_salida_base_humeda_m3_min"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Flujo Seco (Nm³/min)")
    st.plotly_chart(
        px.line(df, y="flujo_gases_salida_base_seca_nm3_min"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Presión
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Presión (atm)")
st.plotly_chart(
    px.line(df, y="presion_gases_salida_atm"),
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)
