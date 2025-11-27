import pandas as pd
import time
import os

# Ruta al dataset de CTGAN
DATA_FILE = "../gan/out_ctgan/datos_sinteticos_ctgan.csv"

# Archivo donde se irán acumulando los datos en tiempo real
STREAM_FILE = "stream_data.csv"

# Cargar datos sintéticos
df = pd.read_csv(DATA_FILE)

# Si existe un archivo previo, se borra para iniciar limpio
if os.path.exists(STREAM_FILE):
    os.remove(STREAM_FILE)

print("Simulador en marcha...")
print(f"Enviando datos cada 10 segundos desde: {DATA_FILE}\n")

i = 0

while True:
    fila = df.iloc[[i]]  # Fila actual

    fila.to_csv(
        STREAM_FILE,
        mode="a",
        header=not os.path.exists(STREAM_FILE),
        index=False
    )

    print(f"[OK] Fila {i} agregada al stream_data.csv")

    i += 1

    if i >= len(df):
        i = 0
        print("\nReiniciando simulación desde el inicio...\n")

    time.sleep(10)
