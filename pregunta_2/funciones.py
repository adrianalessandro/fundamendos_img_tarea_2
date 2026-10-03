"""Funciones base para difusion anisotropica"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np


Coefficient = Callable[[np.ndarray], np.ndarray]


def calcular_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def gradientes(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(u, 1, mode="edge")
    gx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return gx, gy


def laplaciano(u: np.ndarray) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    return (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4 * u
    )


def varianza_local(u: np.ndarray) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    mean = sum(
        padded[i : i + u.shape[0], j : j + u.shape[1]]
        for i in range(3)
        for j in range(3)
    ) / 9
    mean_squared = sum(
        padded[i : i + u.shape[0], j : j + u.shape[1]] ** 2
        for i in range(3)
        for j in range(3)
    ) / 9
    return np.maximum(mean_squared - mean**2, 0)


def difundir(
    u0: np.ndarray,
    coefficient: Coefficient,
    lam: float,
    iterations: int,
    auto_stable: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Difusion explicita con cuatro vecinos y coeficientes en las caras."""
    if lam <= 0 or iterations < 0:
        raise ValueError("lam debe ser positivo e iterations no negativo.")
    u = np.asarray(u0, dtype=float).copy()
    c = coefficient(u)
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("El coeficiente debe ser finito y no negativo.")
    stable = 1 / (4 * max(float(c.max()), np.finfo(float).eps))
    used = min(lam, stable) if auto_stable else lam

    for _ in range(iterations):
        c = coefficient(u)
        padded_u = np.pad(u, 1, mode="edge")
        padded_c = np.pad(c, 1, mode="edge")
        east = 0.5 * (c + padded_c[1:-1, 2:]) * (padded_u[1:-1, 2:] - u)
        west = 0.5 * (c + padded_c[1:-1, :-2]) * (padded_u[1:-1, :-2] - u)
        south = 0.5 * (c + padded_c[2:, 1:-1]) * (padded_u[2:, 1:-1] - u)
        north = 0.5 * (c + padded_c[:-2, 1:-1]) * (padded_u[:-2, 1:-1] - u)
        u = np.clip(u + used * (east + west + south + north), 0, 1)
    return u, coefficient(u), used, stable


def coeficiente_tv(epsilon: float) -> Coefficient:
    if epsilon <= 0:
        raise ValueError("epsilon debe ser positivo.")

    def coefficient(u: np.ndarray) -> np.ndarray:
        gx, gy = gradientes(u)
        return 1 / np.sqrt(gx * gx + gy * gy + epsilon * epsilon)

    return coefficient


def coeficiente_laplaciano(
    k_grad: float = 0.10,
    lap_weight: float = 1.0,
    lap_scale: float = 0.10,
) -> Coefficient:
    def coefficient(u: np.ndarray) -> np.ndarray:
        gx, gy = gradientes(u)
        gradient = np.hypot(gx, gy)
        lap = np.abs(laplaciano(u))
        indicator = (gradient / k_grad) ** 2 + lap_weight * lap / (lap_scale + 1e-12)
        return 1 / (1 + indicator)

    return coefficient


def coeficiente_textura(
    k_grad: float = 0.12,
    k_var: float = 0.025,
    texture_weight: float = 2.0,
) -> Coefficient:
    def coefficient(u: np.ndarray) -> np.ndarray:
        gx, gy = gradientes(u)
        variance = varianza_local(u)
        indicator = (gx * gx + gy * gy) / k_grad**2
        indicator += texture_weight * variance / (k_var + 1e-12)
        return 0.10 + 0.90 / (1 + indicator)

    return coefficient


def cargar_imagen(path: str) -> np.ndarray:
    from PIL import Image

    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    minimum, maximum = image.min(), image.max()
    if maximum == minimum:
        raise ValueError("La imagen no tiene rango dinamico.")
    return (image - minimum) / (maximum - minimum)


def agregar_ruido(
    image: np.ndarray, standard_deviation: float = 0.05, seed: int = 2026
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0, standard_deviation, image.shape), 0, 1)
