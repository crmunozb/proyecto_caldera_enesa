# generar_sinteticos_gan.py
"""
Usa el Generator entrenado para producir datos sintéticos
y guardarlos en un CSV.
"""

from pathlib import Path
import torch
import joblib
import pandas as pd

from gan_modelo import Generator

LATENT_DIM = 32
N_MUESTRAS = 10000  # cámbialo si quieres más/menos muestras


def main():
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "out_gan"

    scaler = joblib.load(out_dir / "scaler.pkl")
    num_cols = joblib.load(out_dir / "num_cols.pkl")

    G = Generator(latent_dim=LATENT_DIM, data_dim=len(num_cols))
    G.load_state_dict(torch.load(out_dir / "generator.pt", map_location="cpu"))
    G.eval()

    z = torch.randn(N_MUESTRAS, LATENT_DIM)
    with torch.no_grad():
        muestras_scaled = G(z).numpy()

    muestras_real = scaler.inverse_transform(muestras_scaled)
    df_fake = pd.DataFrame(muestras_real, columns=num_cols)

    out_csv = out_dir / "datos_sinteticos_gan.csv"
    df_fake.to_csv(out_csv, index=False)
    print(f"[OK] Generadas {N_MUESTRAS} muestras en {out_csv}")


if __name__ == "__main__":
    main()
