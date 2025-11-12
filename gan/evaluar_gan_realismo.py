# evaluar_gan_realismo.py
"""
Evalúa qué tan realistas son los datos sintéticos del GAN comparando:
- Estadísticos univariados + prueba KS
- Matrices de correlación
- Clasificador real vs sintético (AUC)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from gan_datos import cargar_datos


def resumen_univariado(df_real, df_fake, out_dir):
    resumen = []
    cols = df_real.columns

    for col in cols:
        r = df_real[col]
        f = df_fake[col]

        ks_stat, ks_p = ks_2samp(r, f)

        resumen.append(
            {
                "variable": col,
                "real_mean": r.mean(),
                "fake_mean": f.mean(),
                "real_std": r.std(),
                "fake_std": f.std(),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
            }
        )

    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_csv(out_dir / "resumen_univariado.csv", index=False)
    print(df_resumen)
    return df_resumen


def correlacion(df_real, df_fake, out_dir):
    cols = df_real.columns
    corr_real = df_real[cols].corr()
    corr_fake = df_fake[cols].corr()
    diff = corr_real - corr_fake

    corr_real.to_csv(out_dir / "corr_real.csv")
    corr_fake.to_csv(out_dir / "corr_fake.csv")
    diff.to_csv(out_dir / "corr_diff.csv")

    # Heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_real, vmin=-1, vmax=1, annot=False)
    plt.title("Correlación - Datos reales")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_real.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_fake, vmin=-1, vmax=1, annot=False)
    plt.title("Correlación - Datos sintéticos")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_fake.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff, vmin=-1, vmax=1, annot=False, center=0)
    plt.title("Diferencia de correlación (real - sintético)")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_diff.png")
    plt.close()


def clasificador_real_vs_fake(df_real, df_fake, out_dir):
    cols = df_real.columns

    n = min(len(df_real), len(df_fake))
    X_real = df_real[cols].sample(n, random_state=42).values
    X_fake = df_fake[cols].sample(n, random_state=42).values

    X = np.vstack([X_real, X_fake])
    y = np.array([1] * n + [0] * n)  # 1 = real, 0 = fake

    idx_perm = np.random.permutation(len(y))
    X = X[idx_perm]
    y = y[idx_perm]

    n_train = int(0.7 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"AUC clasificador real vs fake: {auc:.3f}")

    with open(out_dir / "clasificador_auc.txt", "w") as f:
        f.write(f"AUC real vs fake: {auc:.3f}\n")


def main():
    this_dir = Path(__file__).resolve().parent
    out_dir = this_dir / "out_gan_eval"
    out_dir.mkdir(exist_ok=True)

    # Datos reales (mismo CSV que usamos para entrenar)
    _, _, df_real = cargar_datos()
    df_real.to_csv(out_dir / "datos_reales_usados.csv", index=False)

    # Datos sintéticos generados previamente
    df_fake = pd.read_csv(this_dir / "out_gan" / "datos_sinteticos_gan.csv")
    # Aseguramos mismas columnas y quitamos NaN
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
