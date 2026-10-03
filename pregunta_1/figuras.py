"""Figuras y exportacion de resultados"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def guardar(figura: plt.Figure, output_dir: Path, nombre: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figura.savefig(output_dir / f"{nombre}.png", dpi=200, bbox_inches="tight")
    plt.close(figura)


def figura_imagenes(x: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    figura, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, image, title in zip(
        axes, (x, y, y - x), ("Imagen ideal x", "Observacion y", "Ruido y - x")
    ):
        im = ax.imshow(
            image,
            cmap="gray",
            vmin=0 if title != "Ruido y - x" else None,
            vmax=1 if title != "Ruido y - x" else None,
        )
        ax.set_title(title)
        ax.set_axis_off()
        figura.colorbar(im, ax=ax, fraction=0.046)
    figura.suptitle("Imagen sintetica y ruido Poisson")
    figura.tight_layout()
    guardar(figura, output_dir, "imagen_sintetica_ruido_poisson")


def figura_kernel(kernel: np.ndarray, filtrada: np.ndarray, sigma: float, output_dir: Path) -> None:
    figura, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(kernel, cmap="magma")
    axes[0].set_title(f"Kernel Gaussiano, sigma={sigma}")
    axes[1].imshow(filtrada, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Resultado filtrado")
    for ax in axes:
        ax.set_axis_off()
    figura.tight_layout()
    guardar(figura, output_dir, "implementacion_kernel_gaussiano")


def figura_rmse(
    sigmas: np.ndarray,
    resultados: dict[str, list[float]],
    optimos: dict[str, tuple[float, float]],
    output_dir: Path,
) -> None:
    colores = {"Fondo (0.15)": "tab:blue", "Cuadrado sin circulo (0.45)": "tab:orange", "Circulo (0.80)": "tab:green"}
    figura, ax = plt.subplots(figsize=(10, 5))
    for nombre, color in colores.items():
        valores = np.asarray(resultados[nombre])
        sigma, error = optimos[nombre]
        ax.plot(sigmas, valores, label=nombre, color=color)
        ax.scatter(sigma, error, color=color, zorder=3)
    sigma_global, _ = optimos["Global"]
    ax.axvline(sigma_global, color="black", linestyle="--", label=f"Minimo global: sigma={sigma_global:.2f}")
    ax.set(xlabel="sigma", ylabel="RMSE", title="RMSE por region segun sigma")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    figura.tight_layout()
    guardar(figura, output_dir, "experimentacion_rmse_sigma")


def figura_perfil(x, y, mejor_global, sigma_global, output_dir: Path) -> None:
    fila = x.shape[0] // 2
    figura, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x[fila], label="Ideal", linewidth=2)
    ax.plot(y[fila], label="Ruidosa", alpha=0.5)
    ax.plot(mejor_global[fila], label=f"Global sigma={sigma_global:.2f}")
    ax.set(xlabel="Columna en fila central", ylabel="Intensidad", title="Perfil central y fronteras")
    ax.legend()
    ax.grid(alpha=0.25)
    figura.tight_layout()
    guardar(figura, output_dir, "experimentacion_perfil_frontera")


def figura_adaptativo(mu_hat, sigma_map, output_dir: Path) -> None:
    figura, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].imshow(mu_hat, cmap="gray", vmin=0, vmax=1)
    im1 = axes[1].imshow(sigma_map, cmap="viridis")
    axes[0].set_title("Estimacion mu_hat")
    axes[1].set_title("Mapa espacial sigma(x,y)")
    for ax in axes:
        ax.set_axis_off()
    figura.colorbar(im0, ax=axes[0], fraction=0.046)
    figura.colorbar(im1, ax=axes[1], fraction=0.046)
    figura.tight_layout()
    guardar(figura, output_dir, "adaptativo_estimacion_muhat_mapa_sigma")


def figura_comparacion(comparacion: dict[str, np.ndarray], output_dir: Path) -> None:
    figura, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (nombre, image) in zip(axes, comparacion.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(nombre)
        ax.set_axis_off()
    figura.tight_layout()
    guardar(figura, output_dir, "adaptativo_comparacion_resultados")


def figura_fronteras(x, mejor_global, adaptativa, sigma_map, output_dir: Path) -> None:
    columna = x.shape[1] // 2
    recorte = slice(52, 76)
    figura, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sigma_map[:, columna], label="sigma(x,y)")
    axes[0].axvline(64, color="k", linestyle=":")
    axes[0].axvline(192, color="k", linestyle=":")
    axes[0].set(xlabel="Fila", ylabel="sigma", title="Transicion del mapa")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x[:, columna][recorte], label="Ideal")
    axes[1].plot(mejor_global[:, columna][recorte], label="Global")
    axes[1].plot(adaptativa[:, columna][recorte], label="Adaptativo")
    axes[1].set(xlabel="Fila (recorte)", ylabel="Intensidad", title="Halo en frontera")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figura.tight_layout()
    guardar(figura, output_dir, "adaptativo_analisis_fronteras")
