# PCA con paralelización en GPU

Este proyecto aplica análisis de componentes principales (PCA) sobre una gran matriz de datos, usando dos enfoques:

- **Versión secuencial (CPU):** utilizando NumPy para procesar los datos en un solo hilo.
- **Versión paralela (GPU + Joblib):** dividiendo la matriz en bloques y calculando autovalores/autovectores en GPU con CuPy, paralelizando con Joblib.

## Cómo ejecutar el código

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Lucarodriguez2025/pca-paralelizacion.git
   cd pca-paralelizacion
> Esta línea fue añadida desde la rama lucas-dev para demostrar el uso de ramas.
