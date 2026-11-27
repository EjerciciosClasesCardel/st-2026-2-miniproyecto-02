"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras de una ejecución en CPU.
"""
import math
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
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
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
HORIZONTE = 12

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
ARCHIVO = "datos/trafico_vehicular_peaje_siberia.csv"
CALENDARIO = "datos/festivos_colombia.csv"
FECHA = "mes"
VALOR = "total"
PERIODO = 12
ESCALA = 10_000
ORDEN = (0, 1, 1)
ORDEN_ESTACIONAL = (1, 0, 1, PERIODO)
NOMBRE_CLASICO = "SARIMAX con festivos"


def preparar(serie):
    """TODO: la misma preparación del miniproyecto 1 (huecos y datos atípicos).

    Valida los huecos y conserva los valores originales de la serie.

    Rechaza series vacías o con NaN para exigir una decisión explícita.
    El criterio 1,5 × IQR se usa como diagnóstico, sin recorte automático.
    """
    if serie.empty:
        raise ValueError("La serie está vacía.")
    if serie.isna().any():
        raise ValueError(
            "La serie tiene meses faltantes. Revise los datos y documente "
            "cómo tratarlos antes de continuar."
        )

    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    atipicos = serie[(serie < limite_inferior) | (serie > limite_superior)]
    print(
        f"Diagnóstico IQR: Q1={q1:.2f}; Q3={q3:.2f}; IQR={iqr:.2f}; "
        f"límites=[{limite_inferior:.2f}, {limite_superior:.2f}]."
    )
    print(f"Rango observado: [{serie.min():.2f}, {serie.max():.2f}].")
    print(
        f"Atípicos según 1,5 × IQR: {len(atipicos)}; "
        "se conservan los valores originales."
    )
    return serie



def modelo_clasico(serie):
    """TODO: devuelva la función modelo(entrenamiento, h) del mejor modelo del
    miniproyecto 1, con los mismos órdenes y la misma transformación.

    Reajusta el SARIMAX del miniproyecto 1 con el calendario de cada origen."""
    fechas = serie.index.append(fechas_siguientes(serie, HORIZONTE))
    x = cargar_serie(CALENDARIO, FECHA, "festivos").to_frame().reindex(fechas)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("El calendario de festivos no cubre los meses requeridos.")
    max_iteraciones = 200

    def pronosticar(entrenamiento, h):
        origen = entrenamiento.index[-1]
        exog_entrenamiento = x.loc[entrenamiento.index]
        exog_futuras = x.loc[fechas_siguientes(entrenamiento, h)]
        ajuste = ARIMA(
            entrenamiento / ESCALA,
            order=ORDEN,
            seasonal_order=ORDEN_ESTACIONAL,
            exog=exog_entrenamiento,
        ).fit(
            method="statespace",
            method_kwargs={"method": "lbfgs", "maxiter": max_iteraciones},
        )
        if not ajuste.mle_retvals.get("converged", False):
            raise RuntimeError(
                f"SARIMAX no convergió en el origen {origen:%Y-%m} "
                f"con un límite de {max_iteraciones} iteraciones."
            )
        pronostico = ajuste.forecast(steps=h, exog=exog_futuras).to_numpy() * ESCALA
        if not np.isfinite(pronostico).all():
            raise ValueError(f"SARIMAX produjo pronósticos no finitos en {origen:%Y-%m}.")
        ULTIMO["modelo"] = ajuste
        return pronostico

    return pronosticar



def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title("Tráfico mensual por el peaje Siberia")
    eje.set_ylabel("Vehículos")
    eje.text(0.02, 0.95, "Categorías III a VII", transform=eje.transAxes, va="top")
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
OBJETIVO = "nivel"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Arma una fila con los doce rezagos, el mes y la media de tres meses previos."""
    X = pd.DataFrame({f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)})
    X["mes"] = z.index.month
    X["media_3"] = z.shift(1).rolling(3).mean()
    return X


def comprobar_sin_fuga(serie):
    """Comprueba cada fila alterando el valor que predice y todo su futuro."""
    original = tabla_supervisada(objetivo(serie))
    for i, mes in enumerate(serie.index):
        alterada = serie.copy()
        alterada.iloc[i:] += 1_000_000
        nueva = tabla_supervisada(objetivo(alterada))
        pd.testing.assert_series_equal(
            original.loc[mes], nueva.loc[mes], check_exact=True,
            obj=f"fila de predictores para {mes:%Y-%m}",
        )
    X, _ = filas_completas(serie)
    print(f"  Prueba de fuga: {len(original)} filas invariantes al cambiar su mes y el futuro.")
    print(f"  Tabla: {len(X)} filas completas, {X.shape[1]} predictores, "
          f"primera fila {X.index[0]:%Y-%m}. Objetivo: {OBJETIVO}.")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Agrega a la historia cada pronóstico antes de calcular el siguiente."""
    fechas = fechas_siguientes(y, h)
    historia = y.copy()
    for mes in fechas:
        historia.loc[mes] = np.nan
        X = tabla_supervisada(objetivo(historia)).iloc[[-1]]
        prediccion = ajustado.predict(X)[0]
        historia.loc[mes] = a_nivel(historia.iloc[:-1], prediccion)
    return historia.loc[fechas].to_numpy()


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


# rejilla con dos a cuatro valores por hiperparámetro.
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [3, 5, 8, 12]}),
    "SVR": (svr, {"C": [1_000, 10_000, 100_000], "epsilon": [100, 1_000, 2_000]}),
    "Random Forest": (bosque, {"max_depth": [4, 8, None], "min_samples_leaf": [1, 3, 5]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1, 0.2]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """Elige cada rejilla por RMSE a un paso, antes del primer origen."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    validacion = TimeSeriesSplit(n_splits=5, test_size=6)
    for numero, (entrena, valida) in enumerate(validacion.split(X), start=1):
        print(f"  Pliegue {numero}: {len(entrena)} filas de entrenamiento hasta "
              f"{X.index[entrena[-1]]:%Y-%m}, valida de "
              f"{X.index[valida[0]]:%Y-%m} a {X.index[valida[-1]]:%Y-%m}.")
    elegidos = {}
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        estimador = fabrica()
        if hasattr(estimador, "steps"):
            paso = estimador.steps[-1][0]
            rejilla = {f"{paso}__{clave}": valores for clave, valores in rejilla.items()}
        busqueda = GridSearchCV(
            estimador, rejilla, cv=validacion, scoring="neg_root_mean_squared_error",
            n_jobs=1, refit=False, error_score="raise",
        )
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        print(f"  {nombre}: {elegidos[nombre]}, RMSE de validación {-busqueda.best_score_:.2f}")
    return elegidos


def graficar_importancia(serie, elegidos):
    """Muestra la importancia de ambos ensambles en el tramo previo al backtest."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, layout="constrained")
    for eje, nombre, color in zip(ejes, ("Random Forest", "XGBoost"), ("#2878A0", "#C27735")):
        fabrica, _ = REJILLAS[nombre]
        ajuste = fabrica(**elegidos[nombre]).fit(X, z)
        importancia = pd.Series(ajuste.feature_importances_, index=X.columns).sort_values()
        eje.barh(importancia.index, importancia, color=color)
        eje.set_title(nombre)
        eje.set_xlabel("Importancia relativa (suma 1)")
        print(f"  Importancia {nombre}: {importancia.nlargest(4).round(4).to_dict()}")
    figura.savefig(FIGURAS / "importancia.png", dpi=140)
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


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
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
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
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
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
