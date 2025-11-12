# gan_datos.py
"""
Módulo de carga y preparación de datos para el GAN.

Usa el archivo 'mediciones_agosto.csv' que está en la raíz del proyecto.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columnas candidatas (las que en teoría nos interesan)
CANDIDATE_COLS = [
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

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "mediciones_agosto.csv"


def cargar_datos(csv_path: Path | str | None = None):
    if csv_path is None:
        csv_path = DEFAULT_CSV

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV en {csv_path}. "
            "Cambia DEFAULT_CSV o pasa la ruta correcta a cargar_datos()."
        )

    df = pd.read_csv(csv_path)

    # Nos quedamos solo con las columnas candidatas que existan
    cols_disponibles = [c for c in CANDIDATE_COLS if c in df.columns]
    if len(cols_disponibles) < 3:
        raise ValueError(
            f"Se encontraron muy pocas columnas esperadas en el CSV.\n"
            f"Candidatas: {CANDIDATE_COLS}\n"
            f"Encontradas: {cols_disponibles}"
        )

    # Convertir a numérico y ver NaN
    df_num = df[cols_disponibles].apply(pd.to_numeric, errors="coerce")

    print("Filas totales en CSV:", len(df))
    print("NaN por columna antes de limpiar:")
    print(df_num.isna().sum())

    # 1) Eliminar columnas que son TODO NaN
    nan_counts = df_num.isna().sum()
    cols_all_nan = [c for c in df_num.columns if nan_counts[c] == len(df_num)]
    if cols_all_nan:
        print("Eliminando columnas completamente vacías:", cols_all_nan)
        df_num = df_num.drop(columns=cols_all_nan)

    # 2) Eliminar filas que aún tengan NaN
    df_num = df_num.dropna()
    print("Filas después de dropna():", len(df_num))
    print("Columnas finales usadas para el GAN:", list(df_num.columns))

    if len(df_num) == 0:
        raise ValueError(
            "Después del filtrado no quedó ninguna fila válida para entrenar."
        )

    X = df_num.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, df_num

