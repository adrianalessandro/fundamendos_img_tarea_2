"""Generacion y exportacion de figuras"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def guardar(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def datos(original, noisy, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, image, title in zip(
        axes, [original, noisy, noisy - original], ["Original", "Ruidosa", "Ruido"]
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0 if title != "Ruido" else -0.2,
            vmax=1 if title != "Ruido" else 0.2,
        )
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, "implementacion_datos_ruido")


def tv_sensibilidad(results, eps_values, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, (epsilon, result) in enumerate(zip(eps_values, results)):
        axes[0, j].imshow(result["cmap"], cmap="magma")
        axes[0, j].set_title(f"cTV eps={epsilon}, max={result['cmap'].max():.1f}")
        axes[1, j].imshow(result["result"], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"RMSE={result['rmse']:.4f}")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_tv_sensibilidad_epsilon")


def estabilidad(results, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, result in zip(axes, results):
        axis.imshow(result["result"], cmap="gray", vmin=0, vmax=1)
        axis.set_title(result["name"] + f"\nRMSE={result['rmse']:.4f}")
        axis.axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_estabilidad")


def coeficientes(results, title, parameters, output_dir: Path, filename: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, result in enumerate(results):
        axes[0, j].imshow(result["cmap"], cmap="magma", vmin=0, vmax=1)
        axes[0, j].set_title(f"{title} {parameters[j]}")
        axes[1, j].imshow(result["result"], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"RMSE={result['rmse']:.4f}")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, filename)


def sensibilidad_laplaciano(c0, c1, l0, l1, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, image, title in zip(
        axes.ravel(),
        [c0, c1, l0, l1],
        ["c original", "c ruidosa", "|Laplaciano| original", "|Laplaciano| ruidosa"],
    ):
        axis.imshow(image, cmap="magma")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_sensibilidad_laplaciano")


def comparacion_final(images, titles, original, final, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_comparacion_final")

    r0, r1, c0, c1 = 80, 170, 80, 170
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(
            image[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Recorte filas {r0}:{r1}, columnas {c0}:{c1}")
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_comparacion_recorte")

    x = np.arange(c0, c1)
    figure, axis = plt.subplots(figsize=(12, 3))
    axis.plot(x, original[r0:r1, c0:c1].mean(0), "k", lw=2, label="Original")
    for result in final:
        axis.plot(
            x,
            result["result"][r0:r1, c0:c1].mean(0),
            label=result["name"],
        )
    axis.set_title("Perfil medio horizontal del recorte")
    axis.set_xlabel("Columna")
    axis.set_ylabel("Intensidad")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    guardar(figure, output_dir, "experimentacion_perfil_recorte")
