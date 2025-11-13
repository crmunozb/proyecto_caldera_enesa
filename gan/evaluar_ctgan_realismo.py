# evaluar_ctgan_realismo.py
"""
Evalúa qué tan realistas son los datos sintéticos generados por CTGAN
comparando:
- Estadísticos univariados + prueba KS
- Matrices de correlación
- Clasificador real vs sintético (AUC)
"""

from pathlib import Path
import pandas as pd

from gan_datos import cargar_datos
from evaluar_gan_realismo import (
    resumen_univariado,
    correlacion,
    clasificador_real_vs_fake,
)


def main():
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "out_ctgan_eval"
    out_dir.mkdir(exist_ok=True)

    # Datos reales (los mismos que se usan para entrenar)
    _, _, df_real = cargar_datos()
    df_real.to_csv(out_dir / "datos_reales_usados.csv", index=False)

    # Datos sintéticos CTGAN
    df_fake = pd.read_csv(this_dir / "out_ctgan" / "datos_sinteticos_ctgan.csv")
    # Aseguramos mismas columnas y quitamos NaN por seguridad
    df_fake = df_fake[df_real.columns].dropna()
    df_fake.to_csv(out_dir / "datos_sinteticos_usados.csv", index=False)

    # 1) Estadística univariada + KS
    resumen_univariado(df_real, df_fake, out_dir)

    # 2) Correlación
    correlacion(df_real, df_fake, out_dir)

    # 3) Clasificador real vs fake
    clasificador_real_vs_fake(df_real, df_fake, out_dir)


if __name__ == "__main__":
    main()
