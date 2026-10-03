"""Ejecucion main
Ejecutar desde la raiz del repo con:
    python pregunta_2/main_p2.py
"""

from pathlib import Path

import numpy as np

from experimentaciones import ejecutar
from figuras import (
    coeficientes,
    comparacion_final,
    datos,
    estabilidad,
    sensibilidad_laplaciano,
    tv_sensibilidad,
)
from funciones import (
    agregar_ruido,
    calcular_rmse,
    cargar_imagen,
    coeficiente_laplaciano,
    coeficiente_textura,
    coeficiente_tv,
    laplaciano,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
IMAGE_PATH = ROOT / "img" / "cameraman.tif"
SEED = 2026
NOISE_STD = 0.05


def main() -> None:
    original = cargar_imagen(str(IMAGE_PATH))
    noisy = agregar_ruido(original, NOISE_STD, SEED)
    datos(original, noisy, OUTPUT_DIR)
    print(f"Tamano: {original.shape}; RMSE ruidosa: {calcular_rmse(noisy, original):.5f}")

    eps_values = [0.005, 0.02, 0.08, 0.20]
    tv_results = [
        ejecutar(f"TV eps={epsilon}", noisy, original, coeficiente_tv(epsilon))
        for epsilon in eps_values
    ]
    tv_sensibilidad(tv_results, eps_values, OUTPUT_DIR)
    for result in tv_results:
        print(result["name"], f"lambda={result['used']:.5f}", f"RMSE={result['rmse']:.5f}")

    cases = [
        ("Lento", coeficiente_tv(0.08), 0.01, 10, True),
        ("Razonable", coeficiente_tv(0.08), 0.20, 80, True),
        ("Sobre-suavizado", coeficiente_tv(0.20), 0.20, 350, True),
        ("Inestable", coeficiente_tv(0.005), 0.20, 40, False),
    ]
    joint = [
        ejecutar(name, noisy, original, coefficient, lam, iterations, auto)
        for name, coefficient, lam, iterations, auto in cases
    ]
    estabilidad(joint, OUTPUT_DIR)
    for result in joint:
        print(
            f"{result['name']}: lambda={result['used']:.5f}, "
            f"estable={result['stable']:.5f}, RMSE={result['rmse']:.5f}, "
            f"rango=[{result['result'].min():.3f},{result['result'].max():.3f}]"
        )

    lap_params = [(0.06, 0.5, 0.08), (0.10, 1.0, 0.10), (0.18, 2.0, 0.15)]
    texture_params = [(0.08, 0.015, 1.0), (0.12, 0.025, 2.0), (0.20, 0.050, 4.0)]
    lap_results = [
        ejecutar(f"Lap A {params}", noisy, original, coeficiente_laplaciano(*params))
        for params in lap_params
    ]
    texture_results = [
        ejecutar(f"Textura B {params}", noisy, original, coeficiente_textura(*params))
        for params in texture_params
    ]
    coeficientes(
        lap_results,
        "Lap A",
        lap_params,
        OUTPUT_DIR,
        "experimentacion_coeficiente_laplaciano",
    )
    coeficientes(
        texture_results,
        "Textura B",
        texture_params,
        OUTPUT_DIR,
        "experimentacion_coeficiente_textura",
    )
    for title, results in [("Lap A", lap_results), ("Textura B", texture_results)]:
        print(title)
        for result in results:
            print(
                f"  {result['name']} RMSE={result['rmse']:.5f} "
                f"max(c)={result['cmap'].max():.4f}"
            )

    lap_coefficient = coeficiente_laplaciano(0.10, 1.0, 0.10)
    c_original, c_noisy = lap_coefficient(original), lap_coefficient(noisy)
    lap_original, lap_noisy = np.abs(laplaciano(original)), np.abs(laplaciano(noisy))
    sensibilidad_laplaciano(
        c_original, c_noisy, lap_original, lap_noisy, OUTPUT_DIR
    )
    print(
        "Mediana |Laplaciano|:",
        f"{np.median(lap_original):.5f} -> {np.median(lap_noisy):.5f}",
        "Percentil 95:",
        f"{np.percentile(lap_original, 95):.5f} -> {np.percentile(lap_noisy, 95):.5f}",
    )

    final = [
        ejecutar("TV eps=.08", noisy, original, coeficiente_tv(0.08), iterations=10),
        ejecutar(
            "Lap A",
            noisy,
            original,
            coeficiente_laplaciano(0.10, 1.0, 0.10),
            iterations=10,
        ),
        ejecutar(
            "Textura B",
            noisy,
            original,
            coeficiente_textura(0.12, 0.025, 2.0),
            iterations=10,
        ),
    ]
    print(f"RMSE ruidosa: {calcular_rmse(noisy, original):.5f}")
    for result in final:
        print(f"{result['name']} RMSE={result['rmse']:.5f}")
    images = [noisy] + [result["result"] for result in final]
    titles = ["Ruidosa"] + [
        result["name"] + f"\nRMSE={result['rmse']:.4f}" for result in final
    ]
    comparacion_final(images, titles, original, final, OUTPUT_DIR)
    print(f"Figuras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()