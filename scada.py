import streamlit as st
import pandas as pd
import time

# ------------------------------
# Configuración del SCADA
# ------------------------------
st.set_page_config(
    page_title="SCADA - Caldera ENESA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# Estilos visuales
# ------------------------------
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #e1e1e1;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            font-size: 40px !important;
        }
        h2, h3, h4 {
            font-weight: 500;
        }
        .card {
            padding: 20px;
            background-color: #161b22;
            border-radius: 12px;
            box-shadow: 1px 1px 8px rgba(255,255,255,0.08);
            margin-bottom: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# TÍTULO
# ------------------------------
st.markdown("<h1>🔥 SCADA – Caldera ENESA (Datos CTGAN)</h1>", unsafe_allow_html=True)

# ------------------------------
# CONTENEDOR QUE SE ACTUALIZA SUAVEMENTE
# ------------------------------
placeholder = st.empty()

# ------------------------------
# LOOP DE ACTUALIZACIÓN (sin refrescar la página)
# ------------------------------
while True:
    with placeholder.container():
        try:
            df = pd.read_csv("stream_data.csv")
        except:
            st.warning("Esperando datos... El archivo stream_data.csv aún no existe.")
            time.sleep(5)
            continue

        # ------------------------------
        # GRÁFICO 1 — Temperatura
        # ------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🌡️ Temperatura de Gases (°C)")
        st.line_chart(df["temperatura_gases_salida_c"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------
        # GRÁFICOS 2 — Oxígeno & Humedad
        # ------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🧪 Oxígeno (%)")
            st.line_chart(df["oxigeno_porcentaje_base_seca"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💧 Humedad (%)")
            st.line_chart(df["humedad_porcentaje"])
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------
        # MP
        # ------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🟤 Material Particulado (mg/m³)")
        st.line_chart(df["concentracion_mp_mg_m3"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------
        # Flujos
        # ------------------------------
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🌪️ Flujo Húmedo (m³/min)")
            st.line_chart(df["flujo_gases_salida_base_humeda_m3_min"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💨 Flujo Seco (Nm³/min)")
            st.line_chart(df["flujo_gases_salida_base_seca_nm3_min"])
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------
        # Presión
        # ------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Presión (atm)")
        st.line_chart(df["presion_gases_salida_atm"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Actualiza cada 5 segundos, pero SIN refrescar la página
    time.sleep(5)
