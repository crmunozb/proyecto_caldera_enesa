# entrenar_gan_sintesis.py
"""
Entrena el GAN sobre los datos reales de la caldera
y guarda Generator, Discriminator y el scaler.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import joblib

from gan_datos import cargar_datos
from gan_modelo import Generator, Discriminator

LATENT_DIM = 32
BATCH_SIZE = 128
EPOCHS = 300
LR = 2e-4


def main():
    # 1) Cargar datos desde el CSV (por defecto mediciones_agosto.csv)
    X_scaled, scaler, df_num = cargar_datos()
    data = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cpu")
    print("Entrenando GAN usando CPU")


    data_dim = data.shape[1]
    G = Generator(latent_dim=LATENT_DIM, data_dim=data_dim).to(device)
    D = Discriminator(data_dim=data_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=LR)
    opt_D = optim.Adam(D.parameters(), lr=LR)

    out_dir = Path(__file__).resolve().parent / "out_gan"
    out_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- Entrenar D ---
            opt_D.zero_grad()

            preds_real = D(real_batch)
            loss_real = criterion(preds_real, real_labels)

            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_batch = G(z).detach()
            preds_fake = D(fake_batch)
            loss_fake = criterion(preds_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # --- Entrenar G ---
            opt_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_batch = G(z)
            preds = D(fake_batch)
            loss_G = criterion(preds, real_labels)
            loss_G.backward()
            opt_G.step()

        if (epoch + 1) % 20 == 0:
            print(
                f"[EPOCH {epoch+1}/{EPOCHS}] "
                f"loss_D={loss_D.item():.4f}  loss_G={loss_G.item():.4f}"
            )

    # Guardar modelos y scaler
    torch.save(G.state_dict(), out_dir / "generator.pt")
    torch.save(D.state_dict(), out_dir / "discriminator.pt")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    # Guardar también los nombres de columnas usadas
    joblib.dump(list(df_num.columns), out_dir / "num_cols.pkl")

    print("[OK] Modelos y scaler guardados en", out_dir)


if __name__ == "__main__":
    main()
