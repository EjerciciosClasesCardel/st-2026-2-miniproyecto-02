"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. Las partes
marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# TODO: los mismos datos del miniproyecto 1: copie la serie a datos/ con su
# ficha en datos/README.md y cambie estos cuatro datos. La serie de ejemplo
# solo sirve para probar que el guion corre.
ARCHIVO = "datos/air_passengers.csv"
FECHA = "Month"
VALOR = "#Passengers"
PERIODO = 12


def preparar(serie):
    """TODO: la misma preparación del miniproyecto 1 (huecos y datos atípicos).

    Debe devolver una serie sin NaN.
    """
    return serie


def modelo_clasico(serie):
    """TODO: devuelva la función modelo(entrenamiento, h) del mejor modelo del
    miniproyecto 1, con los mismos órdenes y la misma transformación.

    Copie aquí lo que esa función necesite, como sus variables exógenas.
    Dentro de modelo, guarde el ajuste en ULTIMO["modelo"] para que la tabla
    cuente sus parámetros.
    """
    raise NotImplementedError("el mejor modelo del miniproyecto 1")


# TODO: el nombre del modelo, con sus órdenes, como aparece en la tabla.
NOMBRE_CLASICO = "SARIMA del miniproyecto 1"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title(f"{VALOR}: serie mensual; a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
OBJETIVO = "nivel"     # TODO: elija "nivel" o "dif12" y justifíquelo
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """TODO: una fila por mes con los rezagos 1 a REZAGOS, al menos el mes del
    año y al menos una estadística de ventana móvil.

    Ninguna columna de la fila t puede usar z(t): los rezagos con shift(k) y
    las ventanas sobre la serie ya corrida un mes.
    """
    raise NotImplementedError("tabla supervisada")


def comprobar_sin_fuga(serie):
    """TODO: pruebe en el código que ninguna fila ve el mes que predice.

    Por ejemplo, cambie el valor de cada mes y verifique que su fila de la
    tabla no cambie. Si una fila cambia, lance AssertionError.
    """
    raise NotImplementedError("prueba de que no hay fuga")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """TODO: pronóstico recursivo a h meses.

    Agregue el mes siguiente a y como NaN, arme la última fila de la tabla,
    pronostique con ajustado.predict, devuelva el pronóstico al nivel con
    a_nivel y escríbalo como si fuera dato antes de pasar al mes siguiente.
    Devuelva los h pronósticos.
    """
    raise NotImplementedError("pronóstico recursivo")


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# TODO: complete cada rejilla con dos a cuatro valores por hiperparámetro.
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [5]}),
    "SVR": (svr, {"C": [1.0], "epsilon": [0.1]}),
    "Random Forest": (bosque, {"max_depth": [None], "min_samples_leaf": [1]}),
    "XGBoost": (xgboost, {"max_depth": [3], "learning_rate": [0.1]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """TODO: para cada modelo de REJILLAS, la combinación con menor RMSE en
    validación temporal.

    Arme la tabla con serie.iloc[:-N_PRUEBA], para que ninguno de los meses
    del backtest decida, y busque con GridSearchCV y TimeSeriesSplit. KNN y
    SVR son make_pipeline: sus claves en la rejilla llevan el prefijo del paso
    (kneighborsregressor__n_neighbors). Imprima lo elegido y su RMSE de
    validación. Devuelva {nombre: {hiperparámetro: valor}}; sin_prefijo
    quita el prefijo.
    """
    raise NotImplementedError("hiperparámetros con validación temporal")


def graficar_importancia(serie, elegidos):
    """TODO: importancia de las variables de Random Forest y XGBoost
    (feature_importances_), ajustados con lo elegido; guarde
    figuras/importancia.png."""
    raise NotImplementedError("importancia de variables")


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32)
        self.red = self.construir(x.shape[-1])
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """TODO: nn.LSTM de una capa sobre la secuencia (lote, 12, 1) y una capa
    lineal que lee el último estado oculto."""

    def __init__(self, ocultas=16):
        super().__init__()
        raise NotImplementedError("LSTM")


# TODO: la tercera red, a elección: Conv1D, un híbrido (el clásico más una
# red sobre sus residuos) o un Transformer pequeño. Una Conv1D o un
# Transformer reciben la secuencia como la LSTM y entran en CANDIDATAS. Un
# híbrido no pasa por con_tabla: escriba su propia función
# modelo(entrenamiento, h) y agregue su fila en main.


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
# TODO: agregue la LSTM, con secuencia=True, y la tercera red.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
}


def validar_redes(serie):
    """TODO: elija las épocas de cada red con validación temporal.

    Arme la tabla con serie.iloc[:-N_PRUEBA] y deje sus últimos 12 meses de
    validación. Entrene cada candidata con Red(...).fit(X_ent, z_ent, X_val,
    z_val): red.perdidas guarda por época la pérdida de entrenamiento y la de
    validación. Grafique las curvas en figuras/perdida.png, quédese con el
    MLP de menor pérdida de validación y devuelva {nombre: épocas}, con las
    épocas de menor pérdida de validación de cada red.
    """
    raise NotImplementedError("validación de las redes")


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """TODO: la mejor red con cinco semillas, cada una por el backtest.

    Use con_tabla(red(nombre, epocas, semilla)). Guarde resultados/semillas.csv
    con una fila por semilla e imprima la media y la desviación del RMSE.
    """
    raise NotImplementedError("cinco semillas")


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio = backtest(serie, modelo)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
