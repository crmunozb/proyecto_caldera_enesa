#!/usr/bin/env python3
import math
import pandas as pd
import numpy as np
import pymysql
from pathlib import Path

# ===== Config =====
XLSX  = "Informe_Agosto.xlsx"
HOJA  = "Hoja1"   # nombre exacto de la hoja

DB = dict(
    host="127.0.0.1",
    port=3306,
    user="enesa_user",
    password="enesapass",
    database="caldera"
)

NUM_COLS = [
    "concentracion_mp_mg_m3","concentracion_mp_sin_corregir_mg_nm3","concentracion_mp_mg_nm3",
    "oxigeno_porcentaje_base_seca","humedad_porcentaje","concentracion_co2_porcentaje",
    "concentracion_co2_sin_corregir_mg_nm3","temperatura_gases_salida_c","presion_gases_salida_atm",
    "flujo_gases_salida_base_humeda_m3_min","flujo_gases_salida_base_seca_nm3_min",
]
CAT_COLS = ["tipo_combustible","combustible","estado_fuente"]

# ===== Utils =====
def norm(s: str) -> str:
    # normaliza encabezados del Excel a snake_case compatible con NUM_COLS/CAT_COLS
    return (
        str(s).strip().lower()
        .replace("\n", " ")
        .replace("  ", " ")
        .replace(" ", "_")
        .replace("/", "_")      # FECHA/HORA -> fecha_hora; MG/M3 -> mg_m3
        .replace("%", "pct")
        .replace("°", "")
        .replace("__", "_")
    )

def a_float(v):
    # convierte strings con miles y coma decimal a float; deja None si no se puede
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return float(v)
    try:
        s = str(v).strip().replace(" ", "").replace(",", ".")
        # elimina separadores de miles tipo 1.234.567,89
        s = pd.Series([s]).str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True).iloc[0]
        return float(s)
    except Exception:
        return None

def clean(v):
    # asegura None (NULL) para MySQL si hay NaN/inf/pandas-NA
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def _to_dt(series: pd.Series) -> pd.Series:
    fh = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if fh.notna().sum() == 0:
        fh = pd.to_datetime(series, dayfirst=False, errors="coerce")
    return fh

def parse_fecha_hora(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)
    # 1) Columnas combinadas
    for c in ["fecha_hora", "fecha_y_hora", "timestamp", "datetime"]:
        if c in cols:
            return _to_dt(df[c])
    # 2) fecha + hora separadas
    if "fecha" in cols and "hora" in cols:
        return _to_dt(df["fecha"].astype(str) + " " + df["hora"].astype(str))
    # 3) solo fecha (incluye serial Excel)
    if "fecha" in cols:
        s = df["fecha"]
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_datetime(s, unit="d", origin="1899-12-30", errors="coerce")
        return _to_dt(s)
    # 4) no encontrada
    raise SystemExit("No encuentro la columna de fecha (fecha_hora / fecha_y_hora / timestamp / datetime / fecha [+ hora]).")

# ===== 1) Leer Excel y normalizar =====
df = pd.read_excel(XLSX, sheet_name=HOJA, header=0, engine="openpyxl")
df.columns = [norm(c) for c in df.columns]

# ===== 2) Parseo de fecha robusto =====
fh = parse_fecha_hora(df)
df["fecha_hora"] = fh
df = df.dropna(subset=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)

# ===== 3) Casting numérico forzado (y dtype object para preservar None) =====
for c in NUM_COLS:
    if c in df.columns:
        df[c] = df[c].apply(a_float).astype(object)

# ===== 4) Insertar a MySQL =====
cols = ["fecha_hora"] + [c for c in (NUM_COLS + CAT_COLS) if c in df.columns]
placeholders = ",".join(["%s"] * len(cols))
sql = f"INSERT INTO mediciones({','.join(cols)}) VALUES ({placeholders})"

con = pymysql.connect(**DB)
cur = con.cursor()

BATCH = 500
batch = []

print(f"[INFO] Iniciando ingesta desde Excel. Filas a insertar: {len(df)}")
print(f"[INFO] Periodo: {df['fecha_hora'].min()} → {df['fecha_hora'].max()}")

for i, r in df.iterrows():
    vals = [pd.to_datetime(r["fecha_hora"]).to_pydatetime()] + [clean(r.get(c, None)) for c in cols[1:]]
    batch.append(vals)
    if len(batch) >= BATCH:
        cur.executemany(sql, batch)
        con.commit()
        print(f"[OK] {i+1}/{len(df)} filas insertadas...")
        batch = []

if batch:
    cur.executemany(sql, batch)
    con.commit()

# ===== 5) Verificación =====
cur.execute("SELECT COUNT(*), MIN(fecha_hora), MAX(fecha_hora) FROM mediciones;")
count, fmin, fmax = cur.fetchone()
print(f"[RESUMEN] Total en BD: {count} | Periodo: {fmin} → {fmax}")

cur.close()
con.close()
print("[LISTO] Ingesta desde Excel completada.")
