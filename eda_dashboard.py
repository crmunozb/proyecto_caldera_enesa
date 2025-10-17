#!/usr/bin/env python3
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go

# =======================
# Configuración & helpers
# =======================
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "enesa_user")
DB_PASS = os.getenv("DB_PASS", "enesa_pass")
DB_NAME = os.getenv("DB_NAME", "caldera")

ENGINE = sa.create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True
)

NUM_COLS = [
    "concentracion_mp_mg_m3",
    "concentracion_mp_sin_corregir_mg_nm3",
    "concentracion_mp_mg_nm3",
    "oxigeno_porcentaje_base_seca",
    "humedad_porcentaje",
    "concentracion_co2_porcentaje",
    "concentracion_co2_sin_corregir_mg_nm3",
    "temperatura_gases_salida_c",
    "presion_gases_salida_atm",
    "flujo_gases_salida_base_humeda_m3_min",
    "flujo_gases_salida_base_seca_nm3_min",
]
CAT_COLS = ["tipo_combustible", "combustible", "estado_fuente"]

# Valores iniciales (semana, fin semiabierto: se suma 1 día internamente)
DEFAULT_START = "2025-09-01"
DEFAULT_END   = "2025-09-07"

def fetch_df(start_date: str, end_date: str, estado=None) -> pd.DataFrame:
    """
    Carga datos entre [start 00:00:00, (end + 1 día) 00:00:00) desde MySQL.
    Si 'estado' viene, filtra por estado_fuente.
    """
    start_ts = f"{start_date} 00:00:00"
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # semiabierto
    end_ts = end_dt.strftime("%Y-%m-%d 00:00:00")

    base_q = """
    SELECT *
    FROM mediciones_v2
    WHERE fecha_hora >= %(start)s AND fecha_hora < %(end)s
    """
    params = {"start": start_ts, "end": end_ts}
    if estado and estado != "Todos":
        base_q += " AND estado_fuente = %(estado)s"
        params["estado"] = estado
    base_q += " ORDER BY fecha_hora;"

    df = pd.read_sql(base_q, ENGINE, params=params)
    if df.empty:
        return df

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def kpi_card(title, value):
    return html.Div(
        children=[
            html.Div(title, style={"color":"#666", "fontWeight":"600"}),
            html.Div(value, style={"fontSize":"20px", "fontWeight":"700"})
        ],
        style={
            "padding":"14px 16px","border":"1px solid #eee","borderRadius":"14px",
            "boxShadow":"0 2px 8px rgba(0,0,0,0.06)","background":"white",
            "display":"flex","flexDirection":"column","gap":"6px","minWidth":"180px"
        }
    )

def compute_gaps(df: pd.DataFrame) -> int:
    """Cuenta minutos faltantes según min/max del rango cargado."""
    if df.empty:
        return 0
    start = df["fecha_hora"].min().floor("min")
    end   = df["fecha_hora"].max().ceil("min")
    expected = int(((end - start).total_seconds() // 60) + 1)
    return max(expected - len(df), 0)

# =======================
# App (tema moderno)
# =======================
app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
])
app.title = "EDA Caldera ENESA — Dashboard"

# Si pones un logo en assets/logo.png, aparece
header_left = html.Div([
    html.Img(src="/assets/logo.png", style={"height":"34px","marginRight":"10px"}),
    html.Div([
        html.H2("EDA Caldera ENESA", style={"margin":"0"}),
        html.Div("Dashboard interactivo: filtros + KPIs + gráficos", style={"color":"#666"})
    ])
], style={"display":"flex","alignItems":"center","gap":"10px"})

app.layout = html.Div([
    # Header
    header_left,

    # Controles
    html.Div([
        html.Div([
            html.Label("Rango de fechas (incluye ambos días)"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed="2020-01-01",
                start_date=DEFAULT_START,
                end_date=DEFAULT_END,
                display_format="YYYY-MM-DD",
                clearable=False
            )
        ], style={"minWidth":"320px"}),

        html.Div([
            html.Label("Estado fuente"),
            dcc.Dropdown(
                options=["Todos","OPERACION","MANTENCION","DETENIDA"],
                value="Todos", id="estado-filter", clearable=False
            )
        ], style={"minWidth":"220px"}),

        html.Div([
            html.Label("Variable (serie / hist / box)"),
            dcc.Dropdown(NUM_COLS, value="concentracion_mp_mg_m3", id="var-select", clearable=False)
        ], style={"minWidth":"320px"}),

        html.Div([
            html.Label("Muestreo (serie)"),
            dcc.Dropdown(
                options=[{"label":l,"value":v} for l,v in [
                    ("Minuto","T"),("5 minutos","5T"),("Hora","h"),("Día","D")
                ]],
                value="h", id="resample", clearable=False
            )
        ], style={"minWidth":"200px"}),

        html.Div([
            html.Button("Descargar CSV (rango filtrado)", id="btn-download", n_clicks=0, style={
                "width":"100%","height":"38px","borderRadius":"10px","border":"1px solid #ddd","cursor":"pointer",
                "background":"#f8f8f8"
            }),
            dcc.Download(id="download-csv")
        ], style={"alignSelf":"end","minWidth":"260px"})
    ], style={"display":"grid","gridTemplateColumns":"repeat(5, minmax(180px, 1fr))","gap":"12px",
              "alignItems":"end","margin":"16px 0"}),

    # KPIs
    html.Div(id="kpi-row", style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}),

    # Serie temporal
    html.Div([dcc.Graph(id="time-series")], style={"marginBottom":"14px"}),

    # Hist & Box
    html.Div([
        html.Div([dcc.Graph(id="hist")], style={"flex":1,"minWidth":"320px"}),
        html.Div([dcc.Graph(id="box")], style={"flex":1,"minWidth":"320px"}),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"14px"}),

    # Correlación
    html.Div([dcc.Graph(id="corr")], style={"marginBottom":"16px"}),

    # Promedio diario
    html.Div([
        html.H3("Promedio diario de la variable seleccionada"),
        dcc.Graph(id="daily-avg")
    ], style={"marginBottom":"24px"}),

    # Tabla
    html.H3("Datos del rango filtrado"),
    dash_table.DataTable(
        id="table",
        page_size=20,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX":"auto"},
        style_cell={"fontFamily":"Inter, system-ui, Arial", "fontSize":"12px", "padding":"6px"},
        style_header={"fontWeight":"bold"}
    ),

    # Data (invisible) para descarga
    dcc.Store(id="store-data")
], style={"maxWidth":"1200px","margin":"20px auto","padding":"0 12px","fontFamily":"Inter, system-ui, Arial"})

# =======================
# Callbacks
# =======================
@callback(
    Output("kpi-row","children"),
    Output("time-series","figure"),
    Output("hist","figure"),
    Output("box","figure"),
    Output("corr","figure"),
    Output("table","columns"),
    Output("table","data"),
    Output("store-data","data"),
    Output("daily-avg","figure"),
    Input("date-range","start_date"),
    Input("date-range","end_date"),
    Input("estado-filter","value"),
    Input("var-select","value"),
    Input("resample","value")
)
def update_dashboard(start_date, end_date, estado, var_name, resample_rule):
    df = fetch_df(start_date, end_date, estado=estado)

    # Sin datos
    if df.empty:
        empty_cols, empty_data = [], []
        empty_fig = go.Figure()
        kpis = [kpi_card("Filas", "0"), kpi_card("Periodo", f"{start_date} → {end_date}")]
        return kpis, empty_fig, empty_fig, empty_fig, empty_fig, empty_cols, empty_data, [], empty_fig

    # KPIs
    filas = len(df)
    periodo = f"{df['fecha_hora'].min()} → {df['fecha_hora'].max()}"
    mp_mean   = df.get("concentracion_mp_mg_m3", pd.Series(dtype=float)).mean()
    o2_mean   = df.get("oxigeno_porcentaje_base_seca", pd.Series(dtype=float)).mean()
    temp_mean = df.get("temperatura_gases_salida_c", pd.Series(dtype=float)).mean()
    gaps      = compute_gaps(df)

    kpis = [
        kpi_card("Filas", f"{filas:,}"),
        kpi_card("Periodo", periodo),
        kpi_card("Minutos faltantes", f"{gaps:,}"),
        kpi_card("MP (mg/m³) Prom.", f"{mp_mean:.2f}" if not np.isnan(mp_mean) else "—"),
        kpi_card("O₂ Base Seca (%) Prom.", f"{o2_mean:.2f}" if not np.isnan(o2_mean) else "—"),
        kpi_card("Temp. gases (°C) Prom.", f"{temp_mean:.2f}" if not np.isnan(temp_mean) else "—"),
    ]

    # Serie temporal (resample)
    ts = df.set_index("fecha_hora").sort_index()
    if var_name in ts.columns:
        ts_agg = ts[var_name].resample(resample_rule).mean()
        fig_ts = px.line(ts_agg, labels={"value": var_name, "fecha_hora":"Tiempo"},
                         title=f"{var_name} — serie temporal ({resample_rule})")
    else:
        fig_ts = go.Figure()

    # Histograma y Boxplot
    fig_hist = px.histogram(df, x=var_name, nbins=40, title=f"Histograma: {var_name}") if var_name in df.columns else go.Figure()
    fig_box  = px.box(df, y=var_name, points=False, title=f"Boxplot: {var_name}") if var_name in df.columns else go.Figure()

    # Correlación
    present = [c for c in NUM_COLS if c in df.columns]
    if len(present) >= 2:
        corr = df[present].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (Pearson)")
    else:
        fig_corr = go.Figure()

    # Promedios diarios
    if var_name in df.columns:
        df_day = df.groupby(df["fecha_hora"].dt.date)[var_name].mean().reset_index()
        fig_day = px.bar(df_day, x="fecha_hora", y=var_name,
                         title=f"Promedio diario de {var_name}",
                         labels={"fecha_hora":"Fecha", var_name:"Promedio"})
    else:
        fig_day = go.Figure()

    # Tabla
    show_cols = ["fecha_hora"] + [c for c in NUM_COLS if c in df.columns] + [c for c in CAT_COLS if c in df.columns]
    show_cols = [c for c in show_cols if c in df.columns]
    table_cols = [{"name": c, "id": c} for c in show_cols]
    table_data = df[show_cols].head(5000).to_dict("records")  # recorte para UI
    store_data = df[show_cols].to_dict("records")             # para descarga

    return kpis, fig_ts, fig_hist, fig_box, fig_corr, table_cols, table_data, store_data, fig_day

# Descarga CSV del rango filtrado
@callback(
    Output("download-csv","data"),
    Input("btn-download","n_clicks"),
    State("store-data","data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, data):
    if not data:
        return None
    tmp = pd.DataFrame(data)
    return dcc.send_data_frame(tmp.to_csv, "mediciones_filtrado.csv", index=False)

# =======================
# Main (Dash 3.x)
# =======================
if __name__ == "__main__":
    # Para demo: deja debug=True hoy, apágalo cuando presentes.
    app.run(debug=True, host="127.0.0.1", port=8050)
