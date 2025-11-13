# entrenar_ctgan.py
"""
Entrena un modelo CTGAN sobre los datos reales de la caldera
usando los mismos datos limpios que gan_datos.cargar_datos()
y guarda el modelo entrenado.
"""

from pathlib import Path
import joblib
from ctgan import CTGAN

from gan_datos import cargar_datos


EPOCHS = 300
BATCH_SIZE = 128


def main():
    # 1) Cargar datos (df_num son los datos numéricos originales, sin escalar)
    X_scaled, scaler, df_num = cargar_datos()

    print("Entrenando CTGAN en CPU con columnas:")
    print(list(df_num.columns))
    print("Filas reales usadas:", len(df_num))

    # CTGAN en CPU (sin usar GPU)
    model = CTGAN(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        generator_dim=(128, 128),
        discriminator_dim=(128, 64),
        verbose=True,
        enable_gpu=False,  # <- esto desactiva la GPU y evita el warning
        pac=1

    )

    # Entrenar
    model.fit(df_num)

    # Guardar modelo
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "out_ctgan"
    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "ctgan_model.pkl"
    joblib.dump(model, model_path)

    cols_path = out_dir / "ctgan_columns.pkl"
    joblib.dump(list(df_num.columns), cols_path)

    print(f"[OK] CTGAN entrenado y guardado en {model_path}")
    print(f"[OK] Columnas usadas guardadas en {cols_path}")


if __name__ == "__main__":
    main()
