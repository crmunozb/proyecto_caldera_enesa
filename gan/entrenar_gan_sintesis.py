# entrenar_gan_sintesis.py
"""
Entrena un WGAN-GP sobre los datos reales de la caldera
y guarda Generator, Discriminator (critic) y el scaler.
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
EPOCHS = 800           
LR = 1e-4
N_CRITIC = 5           # pasos del critic por cada paso del generator
LAMBDA_GP = 10.0       # peso de la gradient penalty


def gradient_penalty(D, real_data, fake_data, device):
    """Calcula la gradient penalty de WGAN-GP."""
    batch_size = real_data.size(0)

    # Interpolación entre real y fake
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    ones = torch.ones_like(d_interpolates, device=device)

    # gradientes de D(interpolates) respecto a interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def main():
    # 1) Cargar datos desde el CSV (por defecto mediciones_agosto.csv)
    X_scaled, scaler, df_num = cargar_datos()
    data = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cpu")  # si tienes GPU: "cuda" si torch.cuda.is_available() else "cpu"
    print("Entrenando WGAN-GP usando", device)

    data_dim = data.shape[1]
    G = Generator(latent_dim=LATENT_DIM, data_dim=data_dim).to(device)
    D = Discriminator(data_dim=data_dim).to(device)

    # Adam con betas típicas para WGAN-GP
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.0, 0.9))

    out_dir = Path(__file__).resolve().parent / "out_gan"
    out_dir.mkdir(exist_ok=True)

    global_step = 0

    for epoch in range(EPOCHS):
        for i, (real_batch,) in enumerate(loader):
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # -----------------------
            #  Entrenar CRITIC (D)
            # -----------------------
            opt_D.zero_grad()

            # Real
            d_real = D(real_batch).view(-1)
            loss_d_real = -d_real.mean()

            # Fake
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_batch = G(z).detach()
            d_fake = D(fake_batch).view(-1)
            loss_d_fake = d_fake.mean()

            # Gradient penalty
            gp = gradient_penalty(D, real_batch, fake_batch, device)

            loss_D = loss_d_real + loss_d_fake + LAMBDA_GP * gp
            loss_D.backward()
            opt_D.step()

            # -----------------------
            #  Entrenar GENERATOR (G)
            #  cada N_CRITIC pasos
            # -----------------------
            global_step += 1
            if global_step % N_CRITIC == 0:
                opt_G.zero_grad()
                z = torch.randn(batch_size, LATENT_DIM, device=device)
                fake_batch = G(z)
                d_fake_for_G = D(fake_batch).view(-1)
                loss_G = -d_fake_for_G.mean()
                loss_G.backward()
                opt_G.step()
            else:
                # Para poder imprimir algo aunque no se haya actualizado G
                loss_G = torch.tensor(0.0)

        # Fin de epoch: imprimir pérdidas aproximadas
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
