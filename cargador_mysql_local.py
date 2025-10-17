#!/usr/bin/env python3
import time, os, pandas as pd, numpy as np, pymysql
from datetime import datetime

CSV = "mediciones_agosto.csv"
DB_CFG = dict(
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

def norm(c): return c.strip().lower().replace("\n"," ").replace("  "," ").replace(" ","_")
def a_float(v):
    if pd.isna(v): return None
    if isinstance(v,(int,float)): return float(v)
    s = str(v).strip().replace(" ","").replace(",",".")
    s = pd.Series([s]).str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))","",regex=True).iloc[0]
    try: return float(s)
    except: return None

# 1) Carga del CSV
df = pd.read_csv(CSV)
df.columns = [norm(c) for c in df.columns]
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)
for c in NUM_COLS:
    if c in df.columns: df[c] = df[c].apply(a_float)

# 2) Conexión a MySQL local (Docker)
con = pymysql.connect(**DB_CFG)
cur = con.cursor()
print(f"[INFO] Iniciando ingesta a MySQL (Docker local) con {len(df)} filas...")

cols = ["fecha_hora"] + [c for c in NUM_COLS+CAT_COLS if c in df.columns]
placeholders = ",".join(["%s"]*len(cols))
sql = f"INSERT INTO mediciones({','.join(cols)}) VALUES ({placeholders})"

# 3) Simulación tipo “minuto a minuto”
batch, B = [], 200
for i, r in df.iterrows():
    vals = [r["fecha_hora"].to_pydatetime()] + [r.get(c, None) for c in cols[1:]]
    batch.append(vals)
    if len(batch) >= B:
        cur.executemany(sql, batch)
        con.commit()
        print(f"[OK] {i+1}/{len(df)} insertadas...")
        batch = []
        time.sleep(0.3)

if batch:
    cur.executemany(sql, batch)
    con.commit()

# 4) Verificación rápida
cur.execute("SELECT COUNT(*) FROM mediciones;")
print("[RESUMEN] Total filas en BD:", cur.fetchone()[0])

cur.execute("SELECT fecha_hora, concentracion_mp_mg_m3 FROM mediciones ORDER BY fecha_hora DESC LIMIT 5;")
for row in cur.fetchall():
    print(row)

cur.close(); con.close()
print("[LISTO] Simulación de ingesta completada.")
