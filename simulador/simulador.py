import pandas as pd
import time
import os

# Ruta base del archivo actual (simulador.py)
BASE = os.path.dirname(os.path.abspath(__file__))

# Ruta correcta al archivo generado por CTGAN
DATA_FILE = os.path.join(BASE, "..", "gan", "out_ctgan", "datos_sinteticos_ctgan.csv")

# Archivo del stream en el mismo directorio del simulador
STREAM_FILE = os.path.join(BASE, "stream_data.csv")

# Verificar que el dataset existe
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"ERROR: No se encontró el dataset en:\n{DATA_FILE}")

# Cargar datos sintéticos
df = pd.read_csv(DATA_FILE)

# Si existe un archivo previo, se borra para iniciar limpio
if os.path.exists(STREAM_FILE):
    os.remove(STREAM_FILE)

print("Simulador en marcha...")
print(f"Enviando datos cada 10 segundos desde:\n{DATA_FILE}\n")

i = 0

while True:
    fila = df.iloc[[i]]  # Fila actual

    fila.to_csv(
        STREAM_FILE,
        mode="a",
        header=not os.path.exists(STREAM_FILE),
        index=False
    )

    print(f"[OK] Fila {i} agregada a stream_data.csv")

    i += 1

    if i >= len(df):
        i = 0
        print("\nReiniciando simulación desde el inicio...\n")

    time.sleep(10)
