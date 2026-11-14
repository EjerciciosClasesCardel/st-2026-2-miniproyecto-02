"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras que produce este guion.

Serie: cabezas de ganado porcino sacrificadas por mes en Colombia, total
nacional (DANE, Encuesta de Sacrificio de Ganado), de 2008-10 a 2026-06. La
misma serie, la misma preparación y el mismo mejor modelo del miniproyecto 1.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
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
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}
# Los pronósticos de cada origen, por modelo, para ver en qué mes falla cada uno.
PRONOSTICOS = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
ARCHIVO = "datos/sacrificio_porcino.csv"
FECHA = "mes"
VALOR = "porcinos"
PERIODO = 12

# El SARIMAX se ajusta sobre serie / ESCALA (miles de cabezas), como en el
# miniproyecto 1; los pronósticos vuelven en cabezas.
ESCALA = 1e3

# Choques de un solo mes, con causa conocida y que no se repiten. Se
# reemplazan por lo que la STL robusta esperaba en ese mes (tendencia +
# estacional). Es la decisión del miniproyecto 1.
CHOQUES = {"2020-04-01": "cuarentena nacional por el COVID-19",
           "2021-05-01": "paro nacional"}


def preparar(serie):
    """La preparación del miniproyecto 1: interpola los huecos (la serie no
    tiene) y reemplaza los choques de CHOQUES por tendencia + estacional de la
    STL robusta, que con robust=True no se deja arrastrar por el propio choque.
    Los dos meses quedan antes del primer origen del backtest (2024-06)."""
    serie = serie.interpolate(method="linear")
    stl = STL(serie, period=PERIODO, robust=True).fit()
    corregida = serie.copy()
    for mes in CHOQUES:
        corregida[mes] = stl.trend[mes] + stl.seasonal[mes]
    return corregida


# ---------- Calendario de Colombia ----------
# En el miniproyecto 1 los festivos salían del paquete holidays. Aquí solo
# entran numpy, pandas, matplotlib, statsmodels, scikit-learn, xgboost y
# torch, así que el calendario se calcula con la ley: la Ley 51 de 1983
# (Ley Emiliani) corre al lunes siguiente los festivos que no caen en lunes.

def pascua(anio):
    """Domingo de Pascua del calendario gregoriano (algoritmo anónimo)."""
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(anio, mes, dia)


def al_lunes(fecha):
    """Ley Emiliani: el festivo pasa al lunes siguiente si no cae en lunes."""
    return fecha + pd.Timedelta(days=(7 - fecha.weekday()) % 7)


_FESTIVOS = {}


def festivos(anio):
    """Los festivos de Colombia de un año, como fechas numpy."""
    if anio not in _FESTIVOS:
        fijos = [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]
        trasladables = [(1, 6), (3, 19), (6, 29), (8, 15), (10, 12), (11, 1), (11, 11)]
        if anio >= 2026:   # Virgen de Chiquinquirá, 9 de julio, festivo desde 2026
            trasladables.append((7, 9))
        p = pascua(anio)
        dias = {pd.Timestamp(anio, m, d) for m, d in fijos}
        dias |= {al_lunes(pd.Timestamp(anio, m, d)) for m, d in trasladables}
        dias |= {p - pd.Timedelta(days=3), p - pd.Timedelta(days=2)}   # jueves y viernes santo
        dias |= {p + pd.Timedelta(days=43), p + pd.Timedelta(days=64),  # Ascensión, Corpus
                 p + pd.Timedelta(days=71)}                             # y Sagrado Corazón
        _FESTIVOS[anio] = np.array(sorted(dias), dtype="datetime64[D]")
    return _FESTIVOS[anio]


_HABILES = {}


def dias_habiles(fechas):
    """Lunes a viernes que no son festivo, en cada mes de `fechas`.

    Es la exógena del SARIMAX del miniproyecto 1 y una columna de la tabla
    supervisada: se conoce con años de anticipación, no hay que pronosticarla.
    """
    valores = []
    for mes in pd.DatetimeIndex(fechas):
        if mes not in _HABILES:
            siguiente = mes + pd.offsets.MonthBegin(1)
            _HABILES[mes] = float(np.busday_count(mes.date(), siguiente.date(),
                                                  holidays=festivos(mes.year)))
        valores.append(_HABILES[mes])
    return pd.Series(valores, index=pd.DatetimeIndex(fechas), name="habiles")


ORDEN = (1, 1, 1)
ESTACIONAL = (1, 1, 1, PERIODO)
NOMBRE_CLASICO = f"SARIMAX{ORDEN}{ESTACIONAL} + días hábiles"


def modelo_clasico(serie):
    """El mejor modelo del miniproyecto 1: SARIMAX(1,1,1)(1,1,1)12 sobre la serie
    en miles de cabezas, con los días hábiles del mes como exógena. Ajusta con
    los días hábiles del entrenamiento y pronostica con los de los h meses que
    siguen al origen, que son calendario."""
    def modelo(entrenamiento, h):
        x = dias_habiles(entrenamiento.index).to_frame()
        ajuste = ARIMA(entrenamiento / ESCALA, exog=x, order=ORDEN,
                       seasonal_order=ESTACIONAL).fit()
        ULTIMO["modelo"] = ajuste
        futuro = dias_habiles(fechas_siguientes(entrenamiento, h)).to_frame()
        return ajuste.forecast(h, exog=futuro).to_numpy() * ESCALA
    return modelo


def graficar_serie(serie, cruda):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy() / 1e3, lw=1.2, label="serie preparada")
    meses = pd.to_datetime(list(CHOQUES))
    eje.scatter(meses, cruda[meses] / 1e3, color="tab:red", zorder=3,
                label="dato original del choque")
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--",
                label="inicio de los meses del backtest")
    eje.set_title("Sacrificio mensual de ganado porcino en Colombia (DANE, ESAG)")
    eje.set_ylabel("miles de cabezas")
    eje.legend(loc="upper left", frameon=False, fontsize=8)
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior. Se elige "dif12": la serie
# se triplica entre 2008 y 2026 y los meses del backtest quedan por encima de
# todo lo que vio el entrenamiento; un árbol no pronostica fuera del rango de
# su objetivo, y la diferencia anual sí queda dentro de ese rango (lo imprime
# justificar_objetivo y lo mide comparar_objetivo).
OBJETIVO = "dif12"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes t, con columnas que solo miran hacia atrás o al calendario:

    - rezago_k = z(t-k), k = 1..12;
    - mes, mes_sen y mes_cos: el mes del año, entero y en el círculo;
    - habiles y dif_habiles: los días hábiles de t y su diferencia con t-12
      (calendario, conocido de antemano);
    - media_3, media_12 y desv_12: media y desviación móviles de z(t-1), ...,
      sobre la serie ya corrida un mes.
    """
    X = pd.DataFrame({f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)},
                     index=z.index)
    mes = z.index.month
    X["mes"] = mes
    X["mes_sen"] = np.sin(2 * np.pi * mes / 12)
    X["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    habiles = dias_habiles(z.index).to_numpy()
    X["habiles"] = habiles
    X["dif_habiles"] = habiles - dias_habiles(z.index - pd.DateOffset(years=1)).to_numpy()
    previa = z.shift(1)
    X["media_3"] = previa.rolling(3).mean()
    X["media_12"] = previa.rolling(12).mean()
    X["desv_12"] = previa.rolling(12).std()
    return X


def comprobar_sin_fuga(serie):
    """Altera el valor de cada mes, uno por uno, y verifica que su fila no cambie.

    Si la fila del mes t cambiara al alterar y(t), alguna columna estaría
    usando el valor que se quiere predecir. Como control, cuenta en cuántos
    casos la alteración sí aparece en alguna fila posterior: la prueba tiene
    que ver los cambios cuando los hay.
    """
    original = tabla_supervisada(objetivo(serie))
    se_ven_despues = 0
    for i, mes in enumerate(serie.index):
        alterada = serie.copy()
        alterada.iloc[i] = alterada.iloc[i] * 3 + 1e6
        nueva = tabla_supervisada(objetivo(alterada))
        if not np.allclose(original.loc[mes], nueva.loc[mes], equal_nan=True):
            raise AssertionError(f"la fila de {mes:%Y-%m} cambia al alterar su propio mes")
        posteriores = ~np.isclose(original.iloc[i + 1:], nueva.iloc[i + 1:], equal_nan=True)
        se_ven_despues += bool(posteriores.any())
    print(f"  sin fuga: {len(serie)} meses alterados uno por uno; ninguna fila cambió en su "
          f"propio mes, y la alteración sí aparece en alguna fila posterior en "
          f"{se_ven_despues} de {len(serie) - 1} casos")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses: agrega el mes siguiente como NaN, arma
    su fila, pronostica el objetivo, lo lleva al nivel y lo escribe como si
    fuera dato; ese valor entra como rezago_1 del mes que sigue."""
    # La última fila solo necesita los 25 meses anteriores: 12 rezagos de la
    # diferencia anual más la ventana de 12 de z(t-1). Se toman 40 por holgura.
    y = y.iloc[-40:].copy()
    pronosticos = []
    for mes in fechas_siguientes(y, h):
        extendida = pd.concat([y, pd.Series([np.nan], index=[mes])])
        fila_mes = tabla_supervisada(objetivo(extendida)).iloc[[-1]]
        z_hat = float(np.asarray(ajustado.predict(fila_mes)).ravel()[0])
        nivel = a_nivel(y, z_hat)
        y = pd.concat([y, pd.Series([nivel], index=[mes])])
        pronosticos.append(nivel)
    return np.array(pronosticos)


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


def justificar_objetivo(serie):
    """El dato que decide OBJETIVO: dónde quedan los meses del backtest frente
    al rango que vio el entrenamiento, en nivel y en diferencia anual."""
    antes, prueba = serie.iloc[:-N_PRUEBA], serie.iloc[-N_PRUEBA:]
    print(f"  nivel: máximo antes del backtest {antes.max():,.0f} ({antes.idxmax():%Y-%m}); "
          f"{(prueba > antes.max()).sum()} de {N_PRUEBA} meses del backtest lo superan")
    d_antes, d_prueba = antes.diff(PERIODO).dropna(), serie.diff(PERIODO).iloc[-N_PRUEBA:]
    dentro = d_prueba.between(d_antes.min(), d_antes.max()).sum()
    print(f"  dif12: rango antes del backtest [{d_antes.min():,.0f}, {d_antes.max():,.0f}]; "
          f"{dentro} de {N_PRUEBA} meses del backtest quedan dentro")


def comparar_objetivo(serie):
    """El mismo Random Forest por el backtest, aprendiendo el nivel y la diferencia."""
    global OBJETIVO
    elegido = OBJETIVO
    resultados = {}
    try:
        for que in ("nivel", "dif12"):
            OBJETIVO = que
            _, promedio = backtest(serie, con_tabla(bosque))
            resultados[que] = promedio["RMSE"]
    finally:
        OBJETIVO = elegido
    print("  Random Forest por defecto en el backtest: "
          + ", ".join(f"{que} RMSE {valor:,.0f}" for que, valor in resultados.items()))


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.
# El SVR además escala el objetivo: su epsilon y su C están pensados para un
# objetivo de orden 1, y la diferencia anual es de decenas de miles de
# cabezas. TransformedTargetRegressor ajusta ese escalador en cada fit.

def knn(n_neighbors=5, weights="uniform"):
    return make_pipeline(StandardScaler(),
                         KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))


def svr(C=1.0, epsilon=0.1):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon)),
        transformer=StandardScaler())


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


REJILLAS = {
    "KNN": (knn, {"n_neighbors": [5, 10, 20, 40], "weights": ["uniform", "distance"]}),
    "SVR": (svr, {"C": [0.3, 1.0, 3.0, 10.0], "epsilon": [0.01, 0.05, 0.1, 0.3]}),
    "Random Forest": (bosque, {"max_depth": [3, 6, None], "min_samples_leaf": [1, 3, 5]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.01, 0.03, 0.1, 0.3]}),
}
PLIEGUES = TimeSeriesSplit(n_splits=5, test_size=12)


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def con_prefijo(estimador, rejilla):
    """La rejilla con las claves que entiende GridSearchCV (kneighborsregressor__k...)."""
    claves = estimador.get_params().keys()
    return {next(c for c in sorted(claves, key=len) if c == k or c.endswith("__" + k)): v
            for k, v in rejilla.items()}


def ajustar_hiperparametros(serie):
    """La combinación con menor RMSE en validación temporal, para cada modelo.

    La tabla se arma con serie.iloc[:-N_PRUEBA]: ninguno de los 24 meses del
    backtest decide. TimeSeriesSplit con 5 pliegues de 12 meses: cada pliegue
    entrena con todo lo anterior y valida con el año que sigue.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    validados = [X.index[v] for _, v in PLIEGUES.split(X)]
    print(f"  tabla de búsqueda: {len(X)} filas, de {X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}; "
          f"5 pliegues de validación de 12 meses, de {validados[0][0]:%Y-%m} a "
          f"{validados[-1][-1]:%Y-%m}")
    elegidos, filas = {}, []
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        estimador = fabrica()
        busqueda = GridSearchCV(estimador, con_prefijo(estimador, rejilla), cv=PLIEGUES,
                                scoring="neg_root_mean_squared_error").fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        combinaciones = len(busqueda.cv_results_["params"])
        filas.append({"modelo": nombre, "combinaciones": combinaciones,
                      "elegido": elegidos[nombre],
                      "RMSE validación": round(-busqueda.best_score_, 0)})
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    tabla.to_csv(RESULTADOS / "hiperparametros.csv", index=False)
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost, ajustados con
    lo elegido sobre la misma tabla de búsqueda (los meses antes del backtest)."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    importancias = pd.DataFrame({
        "Random Forest": bosque(**elegidos["Random Forest"]).fit(X, z).feature_importances_,
        "XGBoost": xgboost(**elegidos["XGBoost"]).fit(X, z).feature_importances_,
    }, index=X.columns)
    importancias.round(4).to_csv(RESULTADOS / "importancia.csv")
    for nombre in importancias:
        primeras = importancias[nombre].nlargest(5)
        print(f"  {nombre}, las 5 que más pesan: "
              + ", ".join(f"{c} {v:.2f}" for c, v in primeras.items()))
    orden = importancias.mean(axis=1).sort_values().index
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for eje, nombre, color in zip(ejes, importancias, ["tab:green", "tab:purple"]):
        eje.barh(orden, importancias.loc[orden, nombre], color=color)
        eje.set_title(f"{nombre}: importancia de cada variable")
        eje.set_xlabel("importancia (suma 1)")
        eje.grid(axis="x", alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)


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
    """nn.LSTM de una capa sobre la secuencia (lote, 12, 1) y una capa lineal
    que lee el último estado oculto, el del rezago 1."""

    def __init__(self, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=ocultas, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        estados, _ = self.lstm(x)
        return self.salida(estados[:, -1, :])


class Conv1D(nn.Module):
    """La tercera red: una convolución 1D sobre los 12 rezagos en orden.

    Cada filtro de ancho 3 mira tres meses consecutivos y se desliza por la
    secuencia con los mismos pesos, así que aprende un patrón local (una
    aceleración o un frenazo de la diferencia anual) con muy pocos
    parámetros; una capa lineal combina las 10 posiciones de cada filtro.
    """

    def __init__(self, filtros=8, ancho=3):
        super().__init__()
        self.conv = nn.Conv1d(1, filtros, kernel_size=ancho)
        self.salida = nn.Linear(filtros * (REZAGOS - ancho + 1), 1)

    def forward(self, x):
        mapas = torch.relu(self.conv(x.transpose(1, 2)))   # (lote, filtros, 10)
        return self.salida(mapas.flatten(1))


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(16), True),
    "Conv1D (8 filtros de 3)": (lambda n: Conv1D(8, 3), True),
}


def validar_redes(serie):
    """Las épocas de cada red con validación temporal anterior al backtest.

    La tabla se arma con serie.iloc[:-N_PRUEBA] y sus últimos 12 meses son la
    validación. Cada candidata se entrena EPOCAS_MAXIMAS épocas guardando las
    dos pérdidas; las épocas elegidas son las de menor pérdida de validación.
    De los dos MLP queda el de menor pérdida de validación.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent, X_val, z_val = X.iloc[:-12], z.iloc[:-12], X.iloc[-12:], z.iloc[-12:]
    print(f"  entrenamiento {X_ent.index[0]:%Y-%m} a {X_ent.index[-1]:%Y-%m} ({len(X_ent)} "
          f"filas), validación {X_val.index[0]:%Y-%m} a {X_val.index[-1]:%Y-%m}")
    constante = np.mean(((z_val - z_ent.mean()) / z_ent.std()) ** 2)
    print(f"  referencia: pronosticar siempre la media del entrenamiento da pérdida de "
          f"validación {constante:.3f}; la diferencia anual promedia {z_ent.mean():,.0f} en "
          f"el entrenamiento (desviación {z_ent.std():,.0f}) y {z_val.mean():,.0f} en la "
          f"validación")
    curvas, filas = {}, []
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        ajustada = Red(construir, secuencia).fit(X_ent, z_ent, X_val, z_val)
        perdidas = np.array(ajustada.perdidas)
        curvas[nombre] = perdidas
        mejor = int(np.argmin(perdidas[:, 1])) + 1
        filas.append({"red": nombre, "parametros": contar_parametros(ajustada),
                      "epocas": mejor, "perdida entrenamiento": perdidas[mejor - 1, 0],
                      "perdida validacion": perdidas[mejor - 1, 1],
                      "validacion en 300": perdidas[-1, 1],
                      "RMSE validacion": np.sqrt(perdidas[mejor - 1, 1]) * z_ent.std()})
    tabla = pd.DataFrame(filas)
    print(tabla.round(3).to_string(index=False))
    tabla.round(4).to_csv(RESULTADOS / "validacion_redes.csv", index=False)

    figura, ejes = plt.subplots(2, 2, figsize=(10, 6.5))
    for eje, (nombre, perdidas), (_, f) in zip(ejes.flat, curvas.items(), tabla.iterrows()):
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 0], label="entrenamiento")
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 1], label="validación (12 meses)")
        eje.axvline(f["epocas"], color="gray", linestyle="--",
                    label=f"mínimo de validación: {f['epocas']} épocas")
        eje.set_title(f"{nombre}, {f['parametros']} parámetros")
        eje.set_xlabel("época")
        eje.set_ylabel("MSE (objetivo escalado)")
        eje.set_ylim(0, max(1.5, perdidas[:, 1].min() * 3))
        eje.legend(fontsize=8)
        eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)

    mlp = tabla[tabla["red"].str.startswith("MLP")].sort_values("perdida validacion").iloc[0]
    print(f"  MLP elegido: {mlp['red']}")
    return {f["red"]: int(f["epocas"]) for _, f in tabla.iterrows()
            if not f["red"].startswith("MLP") or f["red"] == mlp["red"]}


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest completo."""
    filas = []
    for semilla in [SEMILLA + i for i in range(5)]:
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas.append({"semilla": semilla, "RMSE": round(promedio["RMSE"], 2),
                      "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    print(f"  {nombre}, {epocas} épocas, con cinco semillas:")
    print(tabla.to_string(index=False))
    media, desv = tabla["RMSE"].mean(), tabla["RMSE"].std()
    print(f"  RMSE: media {media:,.2f}, desviación {desv:,.2f} "
          f"(de {tabla['RMSE'].min():,.0f} a {tabla['RMSE'].max():,.0f})")
    return media, desv


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
    if isinstance(ajustado, TransformedTargetRegressor):
        ajustado = ajustado.regressor_
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
    registro = []

    def registrado(entrenamiento, h):
        pronostico = np.asarray(modelo(entrenamiento, h))
        registro.append((entrenamiento.index[-1], pronostico))
        return pronostico

    inicio = time.perf_counter()
    _, promedio = backtest(serie, registrado)
    segundos = time.perf_counter() - inicio
    PRONOSTICOS[nombre] = registro
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def analizar_errores(serie, tabla):
    """Dónde falla cada modelo: RMSE por horizonte, y MAE y sesgo por mes del año.

    Cada mes del año aparece en varios orígenes y horizontes; el sesgo es la
    media de real - pronóstico (positivo: el modelo se queda corto).
    """
    mae_mes, sesgo_mes, por_horizonte = {}, {}, {}
    for nombre in tabla["modelo"]:
        filas = [(fecha, paso + 1, serie[fecha] - p)
                 for origen, pronostico in PRONOSTICOS[nombre]
                 for paso, (fecha, p) in enumerate(
                     zip(pd.date_range(origen, periods=13, freq="MS")[1:], pronostico))]
        errores = pd.DataFrame(filas, columns=["fecha", "horizonte", "error"])
        grupos = errores.groupby(errores["fecha"].dt.month)["error"]
        mae_mes[nombre] = grupos.apply(lambda e: e.abs().mean())
        sesgo_mes[nombre] = grupos.mean()
        por_horizonte[nombre] = errores.groupby("horizonte")["error"].apply(
            lambda e: np.sqrt(np.mean(e ** 2)))
    por_horizonte = pd.DataFrame(por_horizonte).T
    por_horizonte.round(0).to_csv(RESULTADOS / "error_por_horizonte.csv")
    print("  RMSE por horizonte (miles de cabezas), todos los pronósticos del backtest:")
    print((por_horizonte[[1, 3, 6, 9, 12]] / 1e3).round(1)
          .set_axis(["h=1", "h=3", "h=6", "h=9", "h=12"], axis=1).to_string())
    mae_mes = pd.DataFrame(mae_mes).T.set_axis(MESES, axis=1)
    sesgo_mes = pd.DataFrame(sesgo_mes).T.set_axis(MESES, axis=1)
    mae_mes.round(0).to_csv(RESULTADOS / "error_por_mes.csv")
    print("  MAE por mes del año (miles de cabezas):")
    print((mae_mes / 1e3).round(1).to_string())
    diciembres = serie[serie.index.month == 12]
    print(f"  último diciembre: {diciembres.iloc[-1]:,.0f} ({diciembres.index[-1]:%Y}) frente a "
          f"{diciembres.iloc[-2]:,.0f} ({diciembres.index[-2]:%Y}), "
          f"{(diciembres.iloc[-1] / diciembres.iloc[-2] - 1) * 100:+.1f} %")
    print("  el peor mes de cada modelo:")
    for nombre, valores in mae_mes.iterrows():
        peor = valores.idxmax()
        print(f"    {nombre}: {peor}, MAE {valores[peor]:,.0f}, "
              f"sesgo {sesgo_mes.loc[nombre, peor]:+,.0f}")

    figura, eje = plt.subplots(figsize=(10, 0.45 * len(mae_mes) + 1.6))
    imagen = eje.imshow(mae_mes / 1e3, cmap="Oranges", aspect="auto")
    eje.set_xticks(range(12), MESES)
    eje.set_yticks(range(len(mae_mes)), mae_mes.index, fontsize=8)
    for i in range(len(mae_mes)):
        for j in range(12):
            eje.text(j, i, f"{mae_mes.iat[i, j] / 1e3:.0f}", ha="center", va="center",
                     fontsize=7)
    eje.set_title("MAE del backtest por mes del año (miles de cabezas)")
    figura.colorbar(imagen, ax=eje, fraction=0.03)
    figura.tight_layout()
    figura.savefig(FIGURAS / "error_por_mes.png", dpi=120)
    plt.close(figura)


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy() / 1e3, color="black",
             label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        print(f"  último origen, {nombre}: RMSE {rmse(real, pronostico):,.0f}")
        eje.plot(real.index, pronostico / 1e3, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):,.0f})")
    eje.set_ylabel("miles de cabezas")
    eje.set_title(f"Pronóstico a 12 meses desde el último origen ({entrenamiento.index[-1]:%Y-%m})")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3)
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
    for mes, causa in CHOQUES.items():
        print(f"  {mes[:7]} ({causa}): {cruda[mes]:,.0f} -> {serie[mes]:,.0f}")
    graficar_serie(serie, cruda)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
        print("  miniproyecto 1: RMSE 25279.41, MAE 20848.26, MAPE 3.60; aquí: "
              f"RMSE {filas[-1]['RMSE']}, MAE {filas[-1]['MAE']}, MAPE {filas[-1]['MAPE']}")
        print(f"  coeficientes en el último origen: "
              f"{ULTIMO['modelo'].params.round(3).to_dict()}")
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        X, z = filas_completas(serie.iloc[:-12])
        print(f"  {X.shape[1]} columnas: {', '.join(X.columns)}")
        print(f"  filas completas en el último origen: {len(X)} "
              f"({X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m})")
        print(f"  OBJETIVO = {OBJETIVO}")
        justificar_objetivo(serie)
        comparar_objetivo(serie)
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
            media, desv = repetir_con_semillas(serie, mejor, epocas[mejor])
            referencia = next(f["RMSE"] for f in filas if f["modelo"] == NOMBRE_CLASICO)
            print(f"  diferencia con el clásico: {media - referencia:+,.2f}, "
                  f"{abs(media - referencia) / desv:.1f} desviaciones entre semillas")
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    filas_ultimo = len(filas_completas(serie.iloc[:-12])[0])
    print(f"  parámetros por fila de entrenamiento en el último origen ({filas_ultimo} filas):")
    for _, f in tabla.iterrows():
        print(f"    {f['modelo']}: {f['parametros'] / filas_ultimo:.2f}")
    analizar_errores(serie, tabla)
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
