"""Ejecucion main
Se puede correr desde la raiz del repo como:

python pregunta_1/main_p1.py

"""

from pathlib import Path
import numpy as np

from experimentaciones import barrer_sigmas, obtener_optimos
from figuras import (
    figura_adaptativo,
    figura_comparacion,
    figura_fronteras,
    figura_imagenes,
    figura_kernel,
    figura_perfil,
    figura_rmse,
)
from funciones import (
    construir_imagen_sintetica,
    construir_kernel_gaussiano,
    construir_mapa_sigma,
    calcular_rmse,
    estimar_media_local,
    filtrar_adaptativo,
    filtrar_gaussiano,
    generar_ruido_poisson,
    medir_regiones,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
NPIX = 256
PHOTONS = 40
SEED = 2714


def main() -> None:
    x, mascaras = construir_imagen_sintetica(NPIX)
    y = generar_ruido_poisson(x, PHOTONS, SEED)
    figura_imagenes(x, y, OUTPUT_DIR)

    sigma_demo = 1.7
    figura_kernel(
        construir_kernel_gaussiano(sigma_demo),
        filtrar_gaussiano(y, sigma_demo),
        sigma_demo,
        OUTPUT_DIR,
    )

    sigmas = np.concatenate(([0.0], np.linspace(0.25, 8.0, 64)))
    resultados, filtradas = barrer_sigmas(x, y, mascaras, sigmas)
    optimos = obtener_optimos(resultados, sigmas)
    figura_rmse(sigmas, resultados, optimos, OUTPUT_DIR)
    sigma_global = optimos["Global"][0]
    mejor_global = filtradas[sigma_global]
    figura_perfil(x, y, mejor_global, sigma_global, OUTPUT_DIR)

    intensidades = np.array([0.15, 0.45, 0.80])
    sigma_regiones = np.array([optimos[nombre][0] for nombre in mascaras])
    mu_hat = estimar_media_local(y, sigma_aux=2.0)
    sigma_map = construir_mapa_sigma(mu_hat, intensidades, sigma_regiones)
    figura_adaptativo(mu_hat, sigma_map, OUTPUT_DIR)

    adaptativa = filtrar_adaptativo(y, sigma_map, sigma_regiones)
    comparacion = {"Sin filtrado": y, "Mejor global": mejor_global, "Adaptativo": adaptativa}
    figura_comparacion(comparacion, OUTPUT_DIR)
    figura_fronteras(x, mejor_global, adaptativa, sigma_map, OUTPUT_DIR)

    print("Optimos por region:")
    for nombre, (sigma, error) in optimos.items():
        print(f"  {nombre}: sigma={sigma:.3f}, RMSE={error:.6f}")
    print("\nComparacion RMSE: global, fondo, cuadrado, circulo")
    for nombre, estimacion in comparacion.items():
        valores = medir_regiones(x, estimacion, mascaras)
        print(f"  {nombre:18s} " + " ".join(f"{valor:.6f}" for valor in valores))
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
