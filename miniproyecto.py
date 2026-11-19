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
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
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
# Los mismos datos del miniproyecto 1: precipitación mensual (mm) de la
# estación CDEA El Topacio, cuenca del río Pance (CVC). Ficha en datos/README.md.
ARCHIVO = "datos/pance_mensual.csv"
FECHA = "mes"
VALOR = "valor"
PERIODO = 12
ETIQUETA = "lluvia (mm)"   # para los ejes de las figuras


def preparar(serie):
    """La misma preparación del miniproyecto 1.

    Huecos: la serie no tiene meses faltantes, fechas repetidas ni códigos de
    faltante (la regla del 80 % de días con dato nunca se activó al construirla).
    Atípicos: se buscan en el residuo de una STL robusta con la regla del IQR,
    para comparar cada mes con lo esperado para ese mes y no con toda la serie.
    Se conservan: son lluvia real o meses registrados con totales diarios, un
    posible artefacto de medición que no se puede corregir sin otra fuente.
    """
    indice = pd.date_range(serie.index.min(), serie.index.max(), freq="MS")
    completa = serie.reindex(indice)
    assert completa.isna().sum() == 0, "faltan meses en la secuencia"
    assert not serie.index.duplicated().any(), "hay fechas repetidas"
    assert (serie >= 0).all(), "hay códigos de faltante (valores negativos)"

    residuo = STL(completa, period=PERIODO, robust=True).fit().resid
    q1, q3 = residuo.quantile([0.25, 0.75])
    bajo, alto = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    atipicos = residuo[(residuo < bajo) | (residuo > alto)]
    print(f"  preparar: {len(completa)} meses seguidos, sin huecos; "
          f"{len(atipicos)} atípicos en el residuo STL "
          f"({', '.join(f'{m:%Y-%m}' for m in atipicos.index)}), conservados")
    return completa


# La lluvia está en cientos de mm: como en el miniproyecto 1, el modelo se
# ajusta con la serie dividida por 100 y el pronóstico se multiplica por 100.
# Es la misma constante para todos los meses, así que no introduce fuga.
ESCALA = 100
ORDEN, ORDEN_ESTACIONAL = (0, 0, 0), (0, 1, 1, PERIODO)


def exogenas(serie, h=12):
    """El número de días de cada mes, desde el primer mes de la serie hasta h
    meses después del último. La lluvia del mes es una suma: un mes más largo
    acumula más. Sale del calendario, así que se conoce para cualquier mes
    futuro y no introduce fuga."""
    meses = pd.date_range(serie.index[0], serie.index[-1] + pd.DateOffset(months=h), freq="MS")
    return pd.DataFrame({"dias_mes": meses.days_in_month.astype(float)}, index=meses)


def modelo_clasico(serie):
    """El mejor modelo del miniproyecto 1: SARIMAX(0,0,0)(0,1,1)12 con el
    número de días del mes como exógena, ajustado sobre serie / ESCALA con la
    clase ARIMA de statsmodels, como en el miniproyecto 1."""
    x = exogenas(serie)

    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, exog=x.loc[entrenamiento.index],
                       order=ORDEN, seasonal_order=ORDEN_ESTACIONAL).fit()
        ULTIMO["modelo"] = ajuste
        futuro = x.loc[fechas_siguientes(entrenamiento, h)]
        return ajuste.forecast(h, exog=futuro).to_numpy() * ESCALA
    return modelo


NOMBRE_CLASICO = "SARIMAX(0,0,0)(0,1,1)12 + días del mes"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title("Lluvia mensual en El Topacio (mm); a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(ETIQUETA)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
# Se eligió "nivel". Con los cuatro modelos por defecto en el backtest, el
# nivel gana por RMSE en tres de cuatro (Random Forest 135.0 frente a 161.4,
# XGBoost 134.1 frente a 164.3, KNN 145.2 frente a 169.0); solo SVR mejora con
# dif12 (152.0 frente a 163.3), porque con la diferencia casi repite el año
# anterior. La diferencia anual de la lluvia es casi ruido (su única memoria
# clara es el rezago 12, que ya trae la tabla), y con el nivel la tabla tiene
# 98 filas antes del backtest en lugar de 86.
OBJETIVO = "nivel"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


VENTANA = 3


def tabla_supervisada(z):
    """Una fila por mes: los rezagos 1 a REZAGOS, el mes del año y la media
    móvil de los VENTANA meses anteriores.

    Ninguna columna de la fila t usa z(t): shift(k) trae el valor de hace k
    meses y la media móvil se calcula sobre la serie ya corrida un mes.
    """
    X = pd.DataFrame({f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)},
                     index=z.index)
    X["mes"] = z.index.month
    X[f"media_{VENTANA}"] = z.shift(1).rolling(VENTANA).mean()
    return X


def comprobar_sin_fuga(serie):
    """Prueba en el código que ninguna fila ve el mes que predice.

    Para cada mes t, cambia y(t) por un valor absurdo y vuelve a armar la
    tabla: la fila t (lo que el modelo ve para pronosticar t) tiene que quedar
    igual. Si cambia, hay fuga y se lanza AssertionError.
    """
    y = serie.copy()
    base = tabla_supervisada(objetivo(y))
    for mes in y.index:
        alterada = y.copy()
        alterada[mes] = alterada[mes] + 10_000
        fila = tabla_supervisada(objetivo(alterada)).loc[mes]
        if not fila.equals(base.loc[mes]):
            cambiadas = list(fila.index[(fila != base.loc[mes]) & fila.notna()])
            raise AssertionError(f"fuga: la fila de {mes:%Y-%m} cambia con su propio valor "
                                 f"(columnas {cambiadas})")
    print(f"  sin fuga: en los {len(y)} meses, cambiar y(t) no cambia la fila de t "
          f"({base.shape[1]} columnas: {', '.join(base.columns[[0, REZAGOS - 1, -2, -1]])}, ...)")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    Agrega el mes siguiente como NaN, arma su fila con la tabla (que solo mira
    meses anteriores), pronostica el objetivo, lo devuelve al nivel con a_nivel
    y lo escribe como si fuera dato antes de pasar al mes siguiente. Así, desde
    el segundo mes, los rezagos más recientes son pronósticos, no datos.
    """
    y = y.copy()
    for _ in range(h):
        y.loc[y.index[-1] + pd.offsets.MonthBegin(1)] = np.nan
        fila = tabla_supervisada(objetivo(y)).iloc[[-1]]
        z_hat = float(np.asarray(ajustado.predict(fila)).ravel()[0])
        y.iloc[-1] = a_nivel(y.iloc[:-1], z_hat)
    return y.iloc[-h:].to_numpy()


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
    eje.set_ylabel(ETIQUETA)
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
