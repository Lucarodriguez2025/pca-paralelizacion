# Optimización Paralela de PCA

Este proyecto implementa un análisis de componentes principales (PCA) sobre una matriz grande de datos simulados, usando dos enfoques distintos:

- **Versión secuencial (CPU):** Usando `NumPy` para procesar los datos en un solo hilo.
- **Versión paralela (GPU + Joblib):** Dividiendo la matriz en bloques y calculando autovalores/autovectores en GPU con `CuPy`, paralelizando con `Joblib`.

---

##  Estructura del Proyecto

- `main.py`: Script principal con la lógica para realizar el PCA en modo secuencial y paralelo.
- `graficos.py`: Script para generar gráficos con `matplotlib` a partir de los resultados.
- `*.png`: Archivos de imagen con los gráficos generados automáticamente.
- `README.md`: Este documento.

---

##  Cómo ejecutar el código

### 1. Clonar el repositorio:

```bash
git clone https://github.com/Lucarodriguez2025/pca-paralelizacion.git
cd pca-paralelizacion
