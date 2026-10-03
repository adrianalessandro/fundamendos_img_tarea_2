"""Barridos y comparaciones cuantitativas"""

from __future__ import annotations
import numpy as np
from funciones import calcular_rmse, filtrar_gaussiano


def barrer_sigmas(
    imagen_ideal: np.ndarray,
    imagen_ruidosa: np.ndarray,
    mascaras: dict[str, np.ndarray],
    sigmas: np.ndarray,
) -> tuple[dict[str, list[float]], dict[float, np.ndarray]]:
    resultados = {nombre: [] for nombre in mascaras}
    resultados["Global"] = []
    filtradas: dict[float, np.ndarray] = {}
    for sigma in sigmas:
        estimacion = (
            imagen_ruidosa if sigma == 0 else filtrar_gaussiano(imagen_ruidosa, sigma)
        )
        filtradas[float(sigma)] = estimacion
        resultados["Global"].append(calcular_rmse(imagen_ideal, estimacion))
        for nombre, mascara in mascaras.items():
            resultados[nombre].append(
                calcular_rmse(imagen_ideal, estimacion, mascara)
            )
    return resultados, filtradas


def obtener_optimos(
    resultados: dict[str, list[float]], sigmas: np.ndarray
) -> dict[str, tuple[float, float]]:
    return {
        nombre: (float(sigmas[indice]), float(valores[indice]))
        for nombre, valores in resultados.items()
        for indice in [int(np.argmin(valores))]
    }
