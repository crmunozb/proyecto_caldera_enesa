#!/usr/bin/env python3
import argparse, pathlib, textwrap, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mysql.connector
from dotenv import load_dotenv

# ---------- Utils ----------
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

CAT_COLS = ["tipo_combustible","combustible","estado_fuente"]

def normalize_cols(cols):
    return [c.strip().lower().replace("\n"," ").replace("  "," ").replace(" ","_") for c in cols]

def a_float(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v,(int,float)):
        return float(v)
    s = str(v).strip().replace(" ","").replace(",",".")
    s = pd.Series([s]).str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))","",regex=True).iloc[0]
    try:
        return float(s)
    except:
        return np.nan

def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="EDA local o desde MySQL para mediciones de caldera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Ejemplos:
          python3 eda_local.py --csv mediciones_agosto.csv --out out_eda
          python3 eda_local.py --from-db --start 2025-09-01 --end 2025-09-07 --out out_eda
        """)
    )
    ap.add_argument("--csv", help="Ruta al CSV local (modo tradicional)")
    ap.add_argument("--from-db", action="store_true", help="Leer los datos directamente desde MySQL usando .env")
    ap.add_argument("--start", default="2025-09-01 00:00:00", help="Fecha inicio (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--end", default="2025-09-07 23:59:00", help="Fecha fin (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--out", default="out_eda", help="Carpeta de salida (figuras y reporte)")
    args = ap.parse_args()

    out_dir  = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1) Fuente de datos ===
    if args.from_db:
        load_dotenv()
        DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
        DB_USER = os.getenv("DB_USER", "enesa_user")
        DB_PASS = os.getenv("DB_PASS", "enesa_pass")
        DB_NAME = os.getenv("DB_NAME", "caldera")
        DB_PORT = int(os.getenv("DB_PORT", "3306"))

        print(f"[INFO] Conectando a MySQL ({DB_HOST}:{DB_PORT}) ...")
        cn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS,
            database=DB_NAME, port=DB_PORT
        )
        query = f"""
        SELECT * FROM mediciones_v2
        WHERE fecha_hora BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY fecha_hora;
        """
        df = pd.read_sql(query, cn)
        cn.close()
        csv_path = pathlib.Path(f"datos_{args.start[:10]}_a_{args.end[:10]}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Se cargaron {len(df)} filas desde MySQL ✅")
    else:
        if not args.csv:
            raise SystemExit("Debes usar --csv o --from-db.")
        csv_path = pathlib.Path(args.csv)
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1")
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, sep=";", encoding="latin-1")
        print(f"[OK] CSV cargado: {csv_path} ({len(df)} filas)")

    # === 2) Limpieza y normalización ===
    df.columns = normalize_cols(df.columns)

    if "fecha_hora" in df.columns:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], dayfirst=True, errors="coerce")
    else:
        candidatos = ["fecha","hora","fecha_y_hora","timestamp","datetime"]
        for c in candidatos:
            if c in df.columns:
                df["fecha_hora"] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
                break
        else:
            raise SystemExit("No encuentro columna 'fecha_hora' (ni alias comunes) en los datos.")
    df = df.dropna(subset=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(a_float)

    # === 3) Resumen y análisis ===
    resumen_txt = out_dir / "resumen.txt"
    with resumen_txt.open("w", encoding="utf-8") as f:
        f.write(f"Fuente: {'MySQL' if args.from_db else csv_path}\n")
        f.write(f"Filas: {len(df):,}\n")
        f.write(f"Periodo: {df['fecha_hora'].min()} -> {df['fecha_hora'].max()}\n\n")
        nulos = df[["fecha_hora", *[c for c in NUM_COLS if c in df.columns]]].isna().sum()
        f.write("Nulos por columna (numéricas y fecha):\n")
        f.write(nulos.to_string())
        f.write("\n\nPrimeras filas:\n")
        f.write(df.head(5).to_string())
    print(f"[OK] Resumen -> {resumen_txt}")

    # === 4) Cobertura temporal ===
    s = df.set_index("fecha_hora").resample("H").size()
    plt.figure(figsize=(10,4))
    plt.plot(s.index, s.values)
    plt.title("Cobertura temporal (registros por hora)")
    plt.xlabel("Tiempo")
    plt.ylabel("Registros/h")
    savefig(out_dir / "01_cobertura_temporal.png")

    # === 5) Histogramas ===
    for c in NUM_COLS:
        if c in df.columns:
            plt.figure(figsize=(6,4))
            plt.hist(df[c].dropna().values, bins=30)
            plt.title(f"Histograma: {c}")
            plt.xlabel(c)
            plt.ylabel("Frecuencia")
            savefig(out_dir / f"hist_{c}.png")

    # === 6) Boxplots ===
    present = [c for c in NUM_COLS if c in df.columns]
    if present:
        plt.figure(figsize=(max(8,len(present)*0.6),4))
        plt.boxplot([df[c].dropna().values for c in present], showfliers=True)
        plt.xticks(range(1,len(present)+1), present, rotation=45, ha="right")
        plt.title("Boxplots variables numéricas")
        savefig(out_dir / "02_boxplots.png")

    # === 7) Correlaciones ===
    if len(present) >= 2:
        corr = df[present].corr(numeric_only=True)
        plt.figure(figsize=(0.8*len(present)+2, 0.8*len(present)+2))
        im = plt.imshow(corr.values, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(present)), present, rotation=45, ha="right")
        plt.yticks(range(len(present)), present)
        plt.title("Matriz de correlaciones (Pearson)")
        savefig(out_dir / "03_correlaciones.png")

    # === 8) Top categorías ===
    for c in CAT_COLS:
        if c in df.columns:
            top = df[c].astype(str).value_counts().head(10)
            plt.figure(figsize=(8,4))
            plt.bar(top.index, top.values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top categorías: {c}")
            plt.ylabel("Frecuencia")
            savefig(out_dir / f"cat_{c}.png")

    # === 9) HTML resumen ===
    html = out_dir / "index.html"
    parts = [
        f"<h2>EDA – {'MySQL' if args.from_db else csv_path.name}</h2>",
        f"<p><b>Filas:</b> {len(df):,} | <b>Periodo:</b> {df['fecha_hora'].min()} → {df['fecha_hora'].max()}</p>",
        "<h3>Cobertura temporal</h3>",
        "<img src='01_cobertura_temporal.png' width='900'>",
        "<h3>Boxplots</h3>",
        "<img src='02_boxplots.png' width='900'>",
        "<h3>Correlaciones</h3>",
        "<img src='03_correlaciones.png' width='900'>",
    ]
    for c in NUM_COLS:
        p = out_dir / f"hist_{c}.png"
        if p.exists():
            parts.append(f"<h3>Histograma: {c}</h3><img src='{p.name}' width='700'>")
    for c in CAT_COLS:
        p = out_dir / f"cat_{c}.png"
        if p.exists():
            parts.append(f"<h3>Categorías: {c}</h3><img src='{p.name}' width='700'>")
    html.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] Figuras y reporte en: {out_dir.resolve()}")
    print(f"Abre en el navegador: {html.resolve()}")

if __name__ == "__main__":
    main()
