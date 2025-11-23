import streamlit as st
import pandas as pd
import plotly.express as px
import time

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
st.set_page_config(
    page_title="SCADA - Caldera ENESA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #e1e1e1;
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

st.markdown("<h1>🔥 SCADA – Caldera ENESA (Datos CTGAN)</h1>", unsafe_allow_html=True)

# Contenedor principal que se actualiza sin recargar la página
placeholder = st.empty()

# ------------------------------
# LOOP SUAVE (sin refrescar página)
# ------------------------------
while True:

    with placeholder.container():

        try:
            df = pd.read_csv("stream_data.csv")
        except:
            st.warning("Esperando datos...")
            time.sleep(5)
            continue

        # ============================================
        #  🌡️ Temperatura
        # ============================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌡️ Temperatura de Gases (°C)")
        fig_temp = px.line(df, y="temperatura_gases_salida_c", title="")
        st.plotly_chart(fig_temp, use_container_width=True, key="chart_temp")
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        #  Oxígeno y Humedad
        # ============================================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧪 Oxígeno (%)")
            fig_o2 = px.line(df, y="oxigeno_porcentaje_base_seca")
            st.plotly_chart(fig_o2, use_container_width=True, key="chart_o2")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("💧 Humedad (%)")
            fig_hum = px.line(df, y="humedad_porcentaje")
            st.plotly_chart(fig_hum, use_container_width=True, key="chart_hum")
            st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # MP
        # ============================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🟤 Material Particulado (mg/m³)")
        fig_mp = px.line(df, y="concentracion_mp_mg_m3")
        st.plotly_chart(fig_mp, use_container_width=True, key="chart_mp")
        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # Flujos
        # ============================================
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🌪️ Flujo Húmedo (m³/min)")
            fig_fh = px.line(df, y="flujo_gases_salida_base_humeda_m3_min")
            st.plotly_chart(fig_fh, use_container_width=True, key="chart_fh")
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("💨 Flujo Seco (Nm³/min)")
            fig_fs = px.line(df, y="flujo_gases_salida_base_seca_nm3_min")
            st.plotly_chart(fig_fs, use_container_width=True, key="chart_fs")
            st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # Presión
        # ============================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Presión (atm)")
        fig_pr = px.line(df, y="presion_gases_salida_atm")
        st.plotly_chart(fig_pr, use_container_width=True, key="chart_pr")
        st.markdown("</div>", unsafe_allow_html=True)

    # Refrescar SOLO los datos, NO recargar página
    time.sleep(10)
