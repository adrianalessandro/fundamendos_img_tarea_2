# Fundamentos de procesamiento de imagenes - Tarea 2

Cada pregunta tiene su notebook inicial que fue usado para los scripts
que reproducen las figuras y los analisis. Cada script en `pregunta_1/` y `pregunta_2/`

## Pregunta 1

Consolidado en:
[`pregunta_1/`](./pregunta_1/).

- [`main_p1.py`](./pregunta_1/main_p1.py): ejecutor principal.
- [`funciones.py`](./pregunta_1/funciones.py): imagen, ruido, kernels, filtros y RMSE.
- [`experimentaciones.py`](./pregunta_1/experimentaciones.py): barrido de sigma.
- [`figuras.py`](./pregunta_1/figuras.py): generacion y exportacion de figuras.
- [`pregunta_1.ipynb`](./pregunta_1.ipynb): informe/notebook.
- [`output/`](./pregunta_1/output/): figuras, outputs.

Para generar los resultados:

```bash
python pregunta_1/main_p1.py
```

## Pregunta 2

Consolidado en:
[`pregunta_2/`](./pregunta_2/).

- [`main_p2.py`](./pregunta_2/main_p2.py): ejecutor principal.
- [`funciones.py`](./pregunta_2/funciones.py): gradientes, Laplaciano,
  difusion explicita y coeficientes `TV`, Laplaciano y textura.
- [`experimentaciones.py`](./pregunta_2/experimentaciones.py): ejecucion
  de cada caso y calculo de RMSE.
- [`figuras.py`](./pregunta_2/figuras.py): generacion y exportacion de figuras.
- [`img/cameraman.tif`](./pregunta_2/img/cameraman.tif): imagen de entrada.
- [`output/`](./pregunta_2/output/): figuras, outputs.

Para regenerar todas las figuras y metricas:

```bash
python pregunta_2/main_p2.py
```

## Modulos dependencias

```bash
pip install numpy scipy matplotlib pillow
```
