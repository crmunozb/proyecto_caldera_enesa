# generar_sinteticos_ctgan.py
"""
Usa el modelo CTGAN entrenado para generar datos sintéticos
y guardarlos en un CSV.
"""

from pathlib import Path
import joblib
import pandas as pd
import torch

N_MUESTRAS = 100000  # cámbialo si quieres más/menos muestras


def main():
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "out_ctgan"
    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "ctgan_model.pkl"
    cols_path = out_dir / "ctgan_columns.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo CTGAN en {model_path}. "
            "Primero ejecuta entrenar_ctgan.py"
        )

    # Cargar modelo y columnas
    model = joblib.load(model_path)
    cols = joblib.load(cols_path)

    # 🔒 FORZAR CTGAN A CPU
    device = torch.device("cpu")
    if hasattr(model, "_device"):
        model._device = device
    if hasattr(model, "_generator"):
        model._generator.to(device)
    if hasattr(model, "_discriminator"):
        model._discriminator.to(device)

    print(f"Generando {N_MUESTRAS} muestras sintéticas con CTGAN en CPU...")

    # Generar muestras sintéticas
    df_fake = model.sample(N_MUESTRAS)

    # Aseguramos el orden de columnas
    df_fake = df_fake[cols]

    out_csv = out_dir / "datos_sinteticos_ctgan.csv"
    df_fake.to_csv(out_csv, index=False)

    print(f"[OK] Generadas {len(df_fake)} muestras sintéticas en {out_csv}")


if __name__ == "__main__":
    main()
