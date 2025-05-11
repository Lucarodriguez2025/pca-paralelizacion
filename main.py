import numpy as np
import cupy as cp
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import time

# Generar una matriz de datos aleatorios normalizados
def generar_datos(n_filas, n_columnas):
    x = np.random.rand(n_filas, n_columnas)
    return StandardScaler().fit_transform(x)

# Autovalores/autovectores en GPU con CuPy
def autovalores_gpu(bloque):
    bloque_gpu = cp.asarray(bloque)
    cov_gpu = cp.cov(bloque_gpu, rowvar=False)
    eigvals, eigvecs = cp.linalg.eigh(cov_gpu)
    return cp.asnumpy(eigvals), cp.asnumpy(eigvecs)

# Autovalores/autovectores en CPU con NumPy
def autovalores_cpu(bloque):
    cov_cpu = np.cov(bloque, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_cpu)
    return eigvals, eigvecs

# Proyección sobre componentes principales
def reducir_dim(X, eigvecs, n_componentes=10):
    return np.dot(X, eigvecs[:, -n_componentes:])

if __name__ == "__main__":
    # Datos simulados
    filas, columnas = 10000, 1000
    X = generar_datos(filas, columnas)

    # División en bloques
    n_bloques = 10
    bloques = np.array_split(X, n_bloques)

    # -------- CÁLCULO PARALELO (GPU + joblib) --------
    t0 = time.time()
    resultados = Parallel(n_jobs=-1)(
        delayed(autovalores_gpu)(bloque) for bloque in bloques
    )
    t1 = time.time()
    print(f"Tiempo total usando CuPy + joblib: {t1 - t0:.4f} segundos")

    _, eigvecs = resultados[0]
    X_reducido = reducir_dim(X, eigvecs, n_componentes=10)
    print(f"Forma final de los datos reducidos: {X_reducido.shape}")

    # -------- CÁLCULO SECUENCIAL (CPU) --------
    t2 = time.time()
    eigvals_cpu, eigvecs_cpu = autovalores_cpu(X)
    t3 = time.time()
    print(f"Tiempo total usando NumPy (secuencial): {t3 - t2:.4f} segundos")

    X_reducido_cpu = reducir_dim(X, eigvecs_cpu, n_componentes=10)
    print(f"Forma final de los datos reducidos (CPU): {X_reducido_cpu.shape}")
