from __future__ import annotations
import numpy as np
from funciones import Coefficient, calcular_rmse, difundir


def ejecutar(
    name: str,
    noisy: np.ndarray,
    original: np.ndarray,
    coefficient: Coefficient,
    lam: float = 0.20,
    iterations: int = 100,
    auto_stable: bool = True,
) -> dict:
    result, cmap, used, stable = difundir(
        noisy, coefficient, lam, iterations, auto_stable
    )
    return {
        "name": name,
        "result": result,
        "cmap": cmap,
        "rmse": calcular_rmse(result, original),
        "used": used,
        "stable": stable,
    }
