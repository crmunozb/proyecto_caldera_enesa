#!/usr/bin/env python3
import os, time, schedule, pymysql, pathlib, logging, argparse, re, math
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Config desde .env / defaults Docker ===
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER", "enesa_user")
DB_PASS = os.getenv("DB_PASS", "enesapass")
DB_NAME = os.getenv("DB_NAME", "caldera")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

# Columnas objetivo en la tabla 'mediciones_v2'
cols_target = [
    "fecha_hora",
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
    "tipo_combustible",
    "combustible",
    "estado_fuente",
]

# Si tus encabezados originales difieren, mapea aquí:
MAPPING = {
    # "FECHA/HORA": "fecha_hora",
    # "CONCENTRACION_MP MG/M3": "concentracion_mp_mg_m3",
    # ...
}

SQL = """
INSERT INTO mediciones_v2 (
  fecha_hora, concentracion_mp_mg_m3, concentracion_mp_sin_corregir_mg_nm3,
  concentracion_mp_mg_nm3, oxigeno_porcentaje_base_seca, humedad_porcentaje,
  concentracion_co2_porcentaje, concentracion_co2_sin_corregir_mg_nm3,
  temperatura_gases_salida_c, presion_gases_salida_atm,
  flujo_gases_salida_base_humeda_m3_min, flujo_gases_salida_base_seca_nm3_min,
  tipo_combustible, combustible, estado_fuente
) VALUES (
  %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
ON DUPLICATE KEY UPDATE
  concentracion_mp_mg_m3=VALUES(concentracion_mp_mg_m3),
  concentracion_mp_sin_corregir_mg_nm3=VALUES(concentracion_mp_sin_corregir_mg_nm3),
  concentracion_mp_mg_nm3=VALUES(concentracion_mp_mg_nm3),
  oxigeno_porcentaje_base_seca=VALUES(oxigeno_porcentaje_base_seca),
  humedad_porcentaje=VALUES(humedad_porcentaje),
  concentracion_co2_porcentaje=VALUES(concentracion_co2_porcentaje),
  concentracion_co2_sin_corregir_mg_nm3=VALUES(concentracion_co2_sin_corregir_mg_nm3),
  temperatura_gases_salida_c=VALUES(temperatura_gases_salida_c),
  presion_gases_salida_atm=VALUES(presion_gases_salida_atm),
  flujo_gases_salida_base_humeda_m3_min=VALUES(flujo_gases_salida_base_humeda_m3_min),
  flujo_gases_salida_base_seca_nm3_min=VALUES(flujo_gases_salida_base_seca_nm3_min),
  tipo_combustible=VALUES(tipo_combustible),
  combustible=VALUES(combustible),
  estado_fuente=VALUES(estado_fuente)
"""

# ---------- Utils ----------
def normalize_cols(cols):
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = (s.replace("\n"," ").replace("  "," ")
               .replace(" ","_").replace("/", "_")
               .replace("%","pct").replace("°",""))
        s = re.sub(r"__+","_", s)
        out.append(s)
    return out

def es_numero(s: str) -> bool:
    s = str(s).strip()
    if not s: return False
    s = s.replace(" ", "").replace(",", ".")
    s = re.sub(r"(?<=\d)\.(?=\d{3}(\D|$))", "", s)
    try:
        float(s); return True
    except:
        return False

def a_float(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v,(int,float)): return float(v)
    s = str(v).strip().replace(" ","").replace(",",".")
    s = re.sub(r"(?<=\d)\.(?=\d{3}(\D|$))","",s)
    try: return float(s)
    except: return None

def parse_fecha(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)
    def _to_dt(x):
        fh = pd.to_datetime(x, dayfirst=True, errors="coerce")
        if fh.notna().sum()==0:
            fh = pd.to_datetime(x, dayfirst=False, errors="coerce")
        return fh

    for c in ["fecha_hora","fecha_y_hora","timestamp","datetime"]:
        if c in cols: return _to_dt(df[c])

    if "fecha" in cols and "hora" in cols:
        return _to_dt(df["fecha"].astype(str)+" "+df["hora"].astype(str))

    if "fecha" in cols:
        s = df["fecha"]
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_datetime(s, unit="d", origin="1899-12-30", errors="coerce")
        return _to_dt(s)

    raise SystemExit("No encuentro columna de fecha (fecha_hora / fecha+hora / timestamp).")

def load_file(path: str, sheet_name=None) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in (".xlsx",".xls"):
        df = pd.read_excel(p, sheet_name=sheet_name or 0, header=0, engine="openpyxl")
    else:
        raise ValueError("Formato no soportado: " + p.suffix)

    df.columns = normalize_cols(df.columns)

    if MAPPING:
        norm_map = {normalize_cols([k])[0]: v for k, v in MAPPING.items()}
        df = df.rename(columns=norm_map)

    # asegura columnas objetivo
    for c in cols_target:
        if c not in df.columns:
            df[c] = None

    # fecha
    fh = parse_fecha(df)
    df["fecha_hora"] = fh
    df = df.dropna(subset=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)

    # numéricos (forzado) y cast a object para preservar None
    for c in cols_target:
        if c != "fecha_hora":
            df[c] = df[c].apply(a_float).astype(object)

    # orden final
    df = df[cols_target].copy()

    # NULLs para PyMySQL (NaN/NaT -> None)
    df = df.astype(object).where(pd.notnull(df), None)
    return df

def validar_fila(row):
    if row["fecha_hora"] is None: return False
    # check rápido en algunas columnas clave
    for col in ["concentracion_mp_mg_m3","oxigeno_porcentaje_base_seca",
                "humedad_porcentaje","temperatura_gases_salida_c","presion_gases_salida_atm"]:
        v = row.get(col)
        if v is not None and not es_numero(v):
            return False
    return True

def obtener_ultima_fecha(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(fecha_hora) FROM mediciones_v2")
            res = cur.fetchone()
            return res[0] if res and res[0] else None
    except Exception:
        return None

# ---------- Tick ----------
def tick(data_path, sheet_name=None):
    try:
        if not pathlib.Path(data_path).exists():
            logging.error(f"No existe el archivo: {data_path}")
            return

        df = load_file(data_path, sheet_name=sheet_name)
        if df.empty:
            logging.info("Archivo sin filas válidas.")
            return

        # Conexión
        conn = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
            database=DB_NAME, charset="utf8mb4", autocommit=True
        )

        # Inserción incremental por última fecha en BD
        ultima_fecha = obtener_ultima_fecha(conn)
        if ultima_fecha:
            df = df[df["fecha_hora"] > ultima_fecha]
            if df.empty:
                logging.info("No hay filas nuevas para cargar.")
                conn.close()
                return

        # Validación fila a fila + batch
        batch = []
        for _, r in df.iterrows():
            if not validar_fila(r):
                continue
            fila = [pd.Timestamp(r["fecha_hora"]).to_pydatetime()]
            for c in cols_target[1:]:
                fila.append(r[c])
            batch.append(tuple(fila))

        if batch:
            with conn.cursor() as cur:
                cur.executemany(SQL, batch)
            logging.info(f"Upsert {len(batch)} filas desde {pathlib.Path(data_path).name}")
        else:
            logging.info("No hay filas válidas que insertar.")

        conn.close()

    except Exception as e:
        logging.error(f"Error en tick: {type(e).__name__}: {e}")

# ---------- Main (scheduler minuto/segundo) ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cargador incremental cada N segundos")
    parser.add_argument("--data", type=str, required=True,
                        help="Ruta a Excel/CSV (ej: Informe_Agosto.xlsx)")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Intervalo en segundos entre ticks (ej: 1, 5, 60)")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Nombre o índice de hoja si es Excel (ej: Hoja1 ó 0)")
    args = parser.parse_args()

    # primera corrida inmediata
    tick(args.data, sheet_name=args.sheet)

    # luego cada N segundos
    logging.info(f"Ingesta corriendo cada {args.interval} s… Ctrl+C para salir")
    schedule.every(args.interval).seconds.do(lambda: tick(args.data, sheet_name=args.sheet))
    while True:
        schedule.run_pending()
        time.sleep(0.5)
