"""Funciones base para la simulacion y el filtrado,
La semilla que uso es 2714"""

from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def construir_imagen_sintetica(
    tamano: int = 256,
    intensidad_fondo: float = 0.15,
    intensidad_cuadrado: float = 0.45,
    intensidad_circulo: float = 0.80,
    lado_cuadrado: int = 128,
    radio_circulo: int = 32,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Construye x y mascaras para las tres regiones"""
    yy, xx = np.indices((tamano, tamano), dtype=float)
    inicio = (tamano - lado_cuadrado) // 2
    fin = inicio + lado_cuadrado
    centro = (tamano - 1) / 2
    cuadrado = (xx >= inicio) & (xx < fin) & (yy >= inicio) & (yy < fin)
    circulo = (xx - centro) ** 2 + (yy - centro) ** 2 <= radio_circulo**2
    circulo &= cuadrado

    imagen = np.full((tamano, tamano), intensidad_fondo, dtype=np.float64)
    imagen[cuadrado] = intensidad_cuadrado
    imagen[circulo] = intensidad_circulo
    mascaras = {
        "Fondo (0.15)": ~cuadrado,
        "Cuadrado sin circulo (0.45)": cuadrado & ~circulo,
        "Circulo (0.80)": circulo,
    }
    if not np.all(sum(mask.astype(int) for mask in mascaras.values()) == 1):
        raise RuntimeError("Las mascaras no son exhaustivas y disjuntas.")
    return imagen, mascaras


def generar_ruido_poisson(
    imagen_ideal: np.ndarray, photons: int = 40, seed: int = 2714
) -> np.ndarray:
    """Aplica y = Poisson(photons*x)/photons"""
    if photons <= 0:
        raise ValueError("photons debe ser positivo.")
    rng = np.random.default_rng(seed)
    return rng.poisson(photons * imagen_ideal) / photons


def construir_kernel_gaussiano(sigma: float) -> np.ndarray:
    """Construye un kernel 2D isotropico normalizado con soporte 3 sigma"""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser un numero real positivo.")
    radio = int(np.ceil(3 * sigma))
    coordenadas = np.arange(-radio, radio + 1, dtype=float)
    X, Y = np.meshgrid(coordenadas, coordenadas, indexing="xy")
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def filtrar_gaussiano(imagen: np.ndarray, sigma: float) -> np.ndarray:
    """Filtra con el kernel definido por construir_kernel_gaussiano"""
    return convolve(imagen, construir_kernel_gaussiano(sigma), mode="reflect")


def calcular_rmse(
    referencia: np.ndarray, estimacion: np.ndarray, mascara: np.ndarray | None = None
) -> float:
    error = referencia - estimacion
    if mascara is not None:
        error = error[mascara]
    return float(np.sqrt(np.mean(error**2)))


def estimar_media_local(imagen: np.ndarray, sigma_aux: float = 2.0) -> np.ndarray:
    """Estima mu_hat con una Gaussiana auxiliar y bordes reflejados"""
    return gaussian_filter(imagen, sigma=sigma_aux, mode="reflect", truncate=3.0)


def construir_mapa_sigma(
    mu_hat: np.ndarray,
    intensidades: np.ndarray,
    sigmas_optimos: np.ndarray,
) -> np.ndarray:
    """Interpola F(mu_hat) y recorta fuera del rango de nodos medidos"""
    orden = np.argsort(intensidades)
    return np.interp(
        mu_hat,
        intensidades[orden],
        sigmas_optimos[orden],
        left=sigmas_optimos[orden][0],
        right=sigmas_optimos[orden][-1],
    )


def filtrar_adaptativo(
    imagen: np.ndarray,
    sigma_map: np.ndarray,
    sigmas_nodales: np.ndarray,
) -> np.ndarray:
    """Aproxima un filtro variable mezclando respuestas de escalas nodales"""
    respuestas = np.stack(
        [filtrar_gaussiano(imagen, sigma) for sigma in sigmas_nodales], axis=0
    )
    posicion = np.interp(sigma_map, sigmas_nodales, np.arange(len(sigmas_nodales)))
    i0 = np.floor(posicion).astype(int)
    i1 = np.minimum(i0 + 1, len(sigmas_nodales) - 1)
    peso = posicion - i0
    filas = np.arange(imagen.shape[0])[:, None]
    columnas = np.arange(imagen.shape[1])[None, :]
    return (1 - peso) * respuestas[i0, filas, columnas] + peso * respuestas[
        i1, filas, columnas
    ]


def medir_regiones(
    referencia: np.ndarray,
    estimacion: np.ndarray,
    mascaras: Mapping[str, np.ndarray],
) -> list[float]:
    return [calcular_rmse(referencia, estimacion)] + [
        calcular_rmse(referencia, estimacion, mascara)
        for mascara in mascaras.values()
    ]
