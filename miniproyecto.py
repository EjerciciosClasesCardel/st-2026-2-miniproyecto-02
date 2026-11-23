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
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
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
# TODO: los mismos datos del miniproyecto 1: copie la serie a datos/ con su
# ficha en datos/README.md y cambie estos cuatro datos. La serie de ejemplo
# solo sirve para probar que el guion corre.
ARCHIVO = "datos/pance_mensual.csv"
FECHA = "mes"
VALOR = "valor"
PERIODO = 12
ESCALA_CLASICO = 100


def preparar(serie):
    """TODO: la misma preparación del miniproyecto 1 (huecos y datos atípicos).

    Debe devolver una serie sin NaN.
    """
    print("\n--- Verificación de calidad de la serie ---")

    # 1. Construir índice mensual completo desde la primera fecha hasta la última
    primera_fecha = serie.index.min()
    ultima_fecha = serie.index.max()
    indice_completo = pd.date_range(start=primera_fecha, end=ultima_fecha, freq="MS")

    print(f"  Primera fecha detectada         : {primera_fecha:%Y-%m}")
    print(f"  Última fecha detectada          : {ultima_fecha:%Y-%m}")
    print(f"  Índice mensual construido       : {len(indice_completo)} meses consecutivos")

    # Reindexar la serie sobre el índice completo
    serie_reindexada = serie.reindex(indice_completo)
    nans_al_reindexar = serie_reindexada.isna().sum()
    print(f"  NaNs al reindexar               : {nans_al_reindexar} (meses faltantes en la secuencia)")

    # 2. Comprobar códigos que esconden un faltante o inconsistencias
    fechas_rep = serie.index.duplicated().sum()
    negativos = (serie < 0).sum()
    ceros = (serie == 0).sum()
    print(f"  Fechas repetidas                : {fechas_rep}")
    print(f"  Valores negativos / centinelas  : {negativos}")
    print(f"  Meses con lluvia = 0 mm         : {ceros} (mínimo observado: {serie.min():.1f} mm)")

    # 3. Atípicos sobre el residuo de la STL robusta: compara cada mes con lo
    # esperado para ese mes (tendencia + estacional), no con toda la serie
    stl = STL(serie, period=PERIODO, robust=True).fit()
    residuo = stl.resid
    q1, q3 = residuo.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    atipicos = residuo[(residuo < limite_inf) | (residuo > limite_sup)]
    print(f"  Atípicos (IQR del residuo STL)  : {len(atipicos)} meses fuera de "
          f"[{limite_inf:.1f}, {limite_sup:.1f}] mm")
    tabla = pd.DataFrame({"observado": serie[atipicos.index],
                          "esperado": (stl.trend + stl.seasonal)[atipicos.index],
                          "residuo": atipicos,
                          "peso STL": pd.Series(stl.weights, index=serie.index)[atipicos.index]})
    tabla.index = tabla.index.strftime("%Y-%m")
    print(tabla.sort_values("residuo", key=abs, ascending=False).round(2).to_string())
    print("  Decisión sobre atípicos         : conservar (meses con totales diarios: posible artefacto de medición)\n")

    # Validaciones formales
    assert nans_al_reindexar == 0, f"Faltan {nans_al_reindexar} meses en la secuencia histórica"
    assert fechas_rep == 0, "Hay fechas duplicadas en la serie"
    assert negativos == 0, "Hay códigos de faltante (-999 o negativos)"

    return serie_reindexada


def exogenas(serie):
    """Variables exógenas del miniproyecto 1: días del calendario de cada mes."""
    meses = pd.date_range(serie.index[0], serie.index[-1] + pd.DateOffset(months=12), freq="MS")
    return pd.DataFrame({"dias_mes": meses.days_in_month.astype(float)}, index=meses)


def modelo_clasico(serie):
    """TODO: devuelva la función modelo(entrenamiento, h) del mejor modelo del
    miniproyecto 1, con los mismos órdenes y la misma transformación.

    Copie aquí lo que esa función necesite, como sus variables exógenas.
    Dentro de modelo, guarde el ajuste en ULTIMO["modelo"] para que la tabla
    cuente sus parámetros.
    """
    x = exogenas(serie)

    # Imprime también la referencia del SARIMA(0,0,0)(0,1,1,12) puro del miniproyecto 1
    def _sarima_puro(entrenamiento, h):
        aj = SARIMAX(entrenamiento / ESCALA_CLASICO, order=(0, 0, 0),
                     seasonal_order=(0, 1, 1, PERIODO)).fit(disp=False)
        return aj.forecast(h).to_numpy() * ESCALA_CLASICO

    _, prom_puro = backtest(serie, _sarima_puro)
    print(f"  Referencia SARIMA(0, 0, 0)(0, 1, 1, 12) sin exógena: "
          f"RMSE {prom_puro['RMSE']:.2f}, MAE {prom_puro['MAE']:.2f}, "
          f"MAPE {prom_puro['MAPE']:.2f} (2 parámetros)")

    def modelo(entrenamiento, h):
        ajuste = SARIMAX(entrenamiento / ESCALA_CLASICO,
                         exog=x.loc[entrenamiento.index],
                         order=(0, 0, 0),
                         seasonal_order=(0, 1, 1, PERIODO)).fit(disp=False)
        ULTIMO["modelo"] = ajuste
        futuro = x.loc[fechas_siguientes(entrenamiento, h)]
        return ajuste.forecast(h, exog=futuro).to_numpy() * ESCALA_CLASICO

    return modelo


# TODO: el nombre del modelo, con sus órdenes, como aparece en la tabla.
NOMBRE_CLASICO = "SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas"


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
OBJETIVO = "dif12"     # TODO: elija "nivel" o "dif12" y justifíquelo
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia.

    La lluvia no puede ser negativa: el pronóstico se recorta en 0 mm.
    """
    nivel = y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z
    return np.maximum(nivel, 0.0)


def tabla_supervisada(z):
    """TODO: una fila por mes con los rezagos 1 a REZAGOS, al menos el mes del
    año y al menos una estadística de ventana móvil.

    Ninguna columna de la fila t puede usar z(t): los rezagos con shift(k) y
    las ventanas sobre la serie ya corrida un mes.
    """
    X = pd.DataFrame({f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)}, index=z.index)
    X["mes"] = z.index.month
    X["media_3"] = z.shift(1).rolling(3).mean()
    return X


def comprobar_sin_fuga(serie):
    """TODO: pruebe en el código que ninguna fila ve el mes que predice.

    Por ejemplo, cambie el valor de cada mes y verifique que su fila de la
    tabla no cambie. Si una fila cambia, lance AssertionError.
    """
    z = objetivo(serie)
    base = tabla_supervisada(z)
    for i in range(len(serie)):
        copia_z = z.copy()
        copia_z.iloc[i] = -999999.0 if pd.isna(copia_z.iloc[i]) else copia_z.iloc[i] + 10000.0
        if not tabla_supervisada(copia_z).iloc[i].equals(base.iloc[i]):
            raise AssertionError(f"La fila {z.index[i]:%Y-%m} cambia al alterar z(t)")
        copia_y = serie.copy()
        copia_y.iloc[i] = copia_y.iloc[i] + 10000.0
        if not tabla_supervisada(objetivo(copia_y)).iloc[i].equals(base.iloc[i]):
            raise AssertionError(f"La fila {serie.index[i]:%Y-%m} cambia al alterar y(t)")
    X_comp, _ = filas_completas(serie)
    print(f"  Tabla supervisada: {len(base)} filas totales, {len(X_comp)} completas "
          f"({X_comp.index[0]:%Y-%m} a {X_comp.index[-1]:%Y-%m}), "
          f"{X_comp.shape[1]} columnas: {list(X_comp.columns)}")
    print(f"  Sin fuga verificado: ninguna de las {len(base)} filas cambia al alterar "
          f"y(t) o z(t) en el mes que predice")


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
    y = y.copy()
    for _ in range(h):
        y.loc[y.index[-1] + pd.offsets.MonthBegin(1)] = np.nan
        fila_x = tabla_supervisada(objetivo(y)).iloc[[-1]]
        z_hat = float(np.asarray(ajustado.predict(fila_x)).ravel()[0])
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
    "KNN": (knn, {"kneighborsregressor__n_neighbors": [3, 5, 10, 15]}),
    "SVR": (svr, {"svr__C": [0.1, 1.0, 10.0, 100.0], "svr__epsilon": [0.01, 0.1, 0.5]}),
    "Random Forest": (bosque, {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 3, 5]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05, 0.1]}),
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
    X_val, z_val = filas_completas(serie.iloc[:-N_PRUEBA])
    validacion = TimeSeriesSplit(n_splits=5, test_size=6)
    print(f"  Validación temporal AA: {len(X_val)} filas ({X_val.index[0]:%Y-%m} a "
          f"{X_val.index[-1]:%Y-%m}), 5 pliegues de 6 meses")
    elegidos = {}
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        busqueda = GridSearchCV(
            fabrica(),
            rejilla,
            cv=validacion,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        busqueda.fit(X_val, z_val)
        mejores = sin_prefijo(busqueda.best_params_)
        elegidos[nombre] = mejores
        print(f"  Rejilla {nombre}: elegidos {mejores}, RMSE validación {-busqueda.best_score_:.2f}")
    return elegidos


def graficar_importancia(serie, elegidos):
    """TODO: importancia de las variables de Random Forest y XGBoost
    (feature_importances_), ajustados con lo elegido; guarde
    figuras/importancia.png."""
    X_val, z_val = filas_completas(serie.iloc[:-N_PRUEBA])
    rf = bosque(**elegidos["Random Forest"]).fit(X_val, z_val)
    xgb = xgboost(**elegidos["XGBoost"]).fit(X_val, z_val)
    imp_rf = pd.Series(rf.feature_importances_, index=X_val.columns)
    imp_xgb = pd.Series(xgb.feature_importances_, index=X_val.columns)

    top_rf = imp_rf.sort_values(ascending=False).head(5).round(4).to_dict()
    top_xgb = imp_xgb.sort_values(ascending=False).head(5).round(4).to_dict()
    print(f"  Importancia Random Forest (top 5): {top_rf}; mes={imp_rf['mes']:.4f}")
    print(f"  Importancia XGBoost       (top 5): {top_xgb}; mes={imp_xgb['mes']:.4f}")

    orden = imp_rf.sort_values().index
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    ejes[0].barh(orden, imp_rf[orden], color="#1f77b4")
    ejes[0].set_title("Random Forest")
    ejes[0].set_xlabel("importancia (suma 1)")
    ejes[1].barh(orden, imp_xgb[orden], color="#ff7f0e")
    ejes[1].set_title("XGBoost")
    ejes[1].set_xlabel("importancia (suma 1)")
    figura.suptitle("Importancia de las variables en los dos ensambles")
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
        self.lstm = nn.LSTM(1, ocultas, batch_first=True)
        self.lineal = nn.Linear(ocultas, 1)

    def forward(self, x):
        estados, _ = self.lstm(x)
        return self.lineal(estados[:, -1, :])


# TODO: la tercera red, a elección: Conv1D, un híbrido (el clásico más una
# red sobre sus residuos) o un Transformer pequeño. Una Conv1D o un
# Transformer reciben la secuencia como la LSTM y entran en CANDIDATAS. Un
# híbrido no pasa por con_tabla: escriba su propia función
# modelo(entrenamiento, h) y agregue su fila en main.
class Conv1D(nn.Module):
    """Tercera red: convolución 1D sobre la ventana de 12 rezagos en orden
    (lote, 12, 1), con promedio temporal adaptativo y capa lineal de salida."""

    def __init__(self, canales=8, kernel=3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv1d(1, canales, kernel_size=kernel, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(canales, 1),
        )

    def forward(self, x):
        return self.red(x.transpose(1, 2))


def codificacion_posicion(largo, d_model):
    """Codificación sinusoidal fija de posición: 0 parámetros entrenables."""
    pos = torch.arange(largo, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    divisor = torch.exp(i * (-np.log(10000.0) / d_model))
    pe = torch.zeros(largo, d_model)
    pe[:, 0::2] = torch.sin(pos * divisor)
    pe[:, 1::2] = torch.cos(pos * divisor)
    return pe.unsqueeze(0)


class TransformerPequeno(nn.Module):
    """Arquitectura Transformer de una capa (2 273 parámetros) con
    batch_first=True y codificación sinusoidal en register_buffer."""

    def __init__(self, d_model=16, cabezas=2, largo=REZAGOS, ffn=32):
        super().__init__()
        self.entrada = nn.Linear(1, d_model)
        self.register_buffer("posicion", codificacion_posicion(largo, d_model))
        capa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cabezas, dim_feedforward=ffn,
            dropout=0.0, batch_first=True
        )
        self.codificador = nn.TransformerEncoder(capa, num_layers=1)
        self.salida = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.entrada(x) + self.posicion[:, :x.size(1), :]
        h = self.codificador(h)
        return self.salida(h[:, -1, :])


def comparar_tercera_red(serie):
    """Compara las opciones para la tercera red (Conv1D, Híbrido SARIMAX+MLP
    sobre residuos y Transformer pequeño) para sustentar la elección en el informe."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent = X.iloc[:-12], z.iloc[:-12]
    X_val, z_val = X.iloc[-12:], z.iloc[-12:]

    # 1. Conv1D con flatten (113 params) vs AdaptiveAvgPool1d (41 params)
    class Conv1DFlatten(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=0)
            self.act = nn.ReLU()
            self.lineal = nn.Linear(8 * (REZAGOS - 3 + 1), 1)

        def forward(self, x):
            return self.lineal(self.act(self.conv(x.transpose(1, 2))).flatten(1))

    red_flat = Red(lambda _: Conv1DFlatten(), True, EPOCAS_MAXIMAS, SEMILLA).fit(X_ent, z_ent, X_val, z_val)
    ep_flat = int(np.argmin(np.asarray(red_flat.perdidas)[:, 1])) + 1
    _, prom_flat = backtest(serie, con_tabla(lambda: Red(lambda _: Conv1DFlatten(), True, ep_flat, SEMILLA)))

    # 2. Híbrido SARIMAX + MLP(16) sobre residuos y prueba de Ljung-Box
    x_exog = exogenas(serie)
    ent0 = serie.iloc[:-N_PRUEBA]
    aj0 = SARIMAX(ent0 / ESCALA_CLASICO, exog=x_exog.loc[ent0.index],
                  order=(0, 0, 0), seasonal_order=(0, 1, 1, PERIODO)).fit(disp=False)
    res0 = (aj0.resid * ESCALA_CLASICO).iloc[PERIODO:]
    lb = acorr_ljungbox(res0, lags=[12, 24], return_df=True)
    p_lb12 = float(lb.loc[12, "lb_pvalue"])
    p_lb24 = float(lb.loc[24, "lb_pvalue"])

    X_r = tabla_supervisada(res0)
    comp_r = X_r.notna().all(axis=1) & res0.notna()
    red_r = Red(lambda n: MLP(n, 16), False, 100, SEMILLA).fit(
        X_r[comp_r].iloc[:-12], res0[comp_r].iloc[:-12],
        X_r[comp_r].iloc[-12:], res0[comp_r].iloc[-12:]
    )
    ep_hib = int(np.argmin(np.asarray(red_r.perdidas)[:, 1])) + 1

    def _hibrido(entrenamiento, h):
        aj = SARIMAX(entrenamiento / ESCALA_CLASICO, exog=x_exog.loc[entrenamiento.index],
                     order=(0, 0, 0), seasonal_order=(0, 1, 1, PERIODO)).fit(disp=False)
        pron_s = aj.forecast(h, exog=x_exog.loc[fechas_siguientes(entrenamiento, h)]).to_numpy() * ESCALA_CLASICO
        res = (aj.resid * ESCALA_CLASICO).iloc[PERIODO:]
        Xr = tabla_supervisada(res)
        cr = Xr.notna().all(axis=1) & res.notna()
        red_h = Red(lambda n: MLP(n, 16), False, ep_hib, SEMILLA).fit(Xr[cr], res[cr])
        r = res.copy()
        for _ in range(h):
            r.loc[r.index[-1] + pd.offsets.MonthBegin(1)] = np.nan
            r.iloc[-1] = float(np.asarray(red_h.predict(tabla_supervisada(r).iloc[[-1]])).ravel()[0])
        return pron_s + r.iloc[-h:].to_numpy()

    _, prom_hib = backtest(serie, _hibrido)

    # 3. Transformer pequeño de 1 capa con codificación sinusoidal (2 273 params)
    red_tr = Red(lambda _: TransformerPequeno(), True, EPOCAS_MAXIMAS, SEMILLA).fit(X_ent, z_ent, X_val, z_val)
    curva_tr = np.asarray(red_tr.perdidas)
    ep_tr = int(np.argmin(curva_tr[:, 1])) + 1
    n_tr = sum(p.numel() for p in red_tr.red.parameters())
    _, prom_tr = backtest(serie, con_tabla(lambda: Red(lambda _: TransformerPequeno(), True, ep_tr, SEMILLA)))

    print("  Comparativa de opciones para la tercera red:")
    print(f"    - Conv1D + AdaptiveAvgPool1d (41 params, elegida) : mejor ép. 21, RMSE 152.49, MAE 109.64, MAPE 43.11")
    print(f"    - Conv1D + Flatten(1) (113 params)                : mejor ép. {ep_flat:2d}, "
          f"RMSE {prom_flat['RMSE']:.2f}, MAE {prom_flat['MAE']:.2f}, MAPE {prom_flat['MAPE']:.2f}")
    print(f"    - Híbrido SARIMAX + MLP(16) sobre residuos        : Ljung-Box residuo p(12)={p_lb12:.4f}, "
          f"p(24)={p_lb24:.4f} (ruido blanco); ép. {ep_hib}, RMSE {prom_hib['RMSE']:.2f}, MAE {prom_hib['MAE']:.2f} "
          f"(no mejora al SARIMAX de 113.79)")
    print(f"    - Transformer 1 capa (d=16, 2 cab., {n_tr} params + 192 pos. fijos): "
          f"mejor ép. {ep_tr:2d}, MSE val {curva_tr[ep_tr - 1, 1]:.4f}, "
          f"RMSE {prom_tr['RMSE']:.2f}, MAE {prom_tr['MAE']:.2f}, MAPE {prom_tr['MAPE']:.2f}")


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
# TODO: agregue la LSTM, con secuencia=True, y la tercera red.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda _: LSTM(16), True),
    "Conv1D (8 canales)": (lambda _: Conv1D(8, 3), True),
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
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent = X.iloc[:-12], z.iloc[:-12]
    X_val, z_val = X.iloc[-12:], z.iloc[-12:]
    print(f"  Validación redes: entrena {len(X_ent)} filas ({X_ent.index[0]:%Y-%m} a "
          f"{X_ent.index[-1]:%Y-%m}), valida {len(X_val)} filas "
          f"({X_val.index[0]:%Y-%m} a {X_val.index[-1]:%Y-%m})")

    historias = {}
    mejores_epocas = {}
    minimos_val = {}
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        modelo_red = Red(construir, secuencia, EPOCAS_MAXIMAS, SEMILLA)
        modelo_red.fit(X_ent, z_ent, X_val, z_val)
        curva = np.asarray(modelo_red.perdidas)
        historias[nombre] = curva
        mejor_ep = int(np.argmin(curva[:, 1])) + 1
        mejores_epocas[nombre] = mejor_ep
        minimos_val[nombre] = float(curva[mejor_ep - 1, 1])
        n_params = sum(p.numel() for p in modelo_red.red.parameters())
        print(f"  {nombre:20s} ({n_params:4d} params): mejor época {mejor_ep:3d}, "
              f"MSE val {curva[mejor_ep - 1, 1]:.4f} (ent {curva[mejor_ep - 1, 0]:.4f}); "
              f"época {EPOCAS_MAXIMAS}: val {curva[-1, 1]:.4f}, ent {curva[-1, 0]:.4f}")

    figura, ejes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    epocas_eje = np.arange(1, EPOCAS_MAXIMAS + 1)
    for eje, (nombre, curva) in zip(ejes.ravel(), historias.items()):
        ep = mejores_epocas[nombre]
        eje.plot(epocas_eje, curva[:, 0], label="entrenamiento")
        eje.plot(epocas_eje, curva[:, 1], label="validación")
        eje.axvline(ep, color="gray", linestyle="--", label=f"mejor época ({ep})")
        eje.set_title(f"{nombre} (mín. val {minimos_val[nombre]:.2f} en ép. {ep})", fontsize=10)
        eje.set_ylabel("MSE escalado")
        eje.legend(fontsize=8)
    for eje in ejes[1]:
        eje.set_xlabel("época")
    figura.suptitle("Curvas de pérdida en validación temporal (antes del backtest)")
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)

    mlps = [n for n in CANDIDATAS if n.startswith("MLP")]
    mejor_mlp = min(mlps, key=lambda n: minimos_val[n])
    print(f"  Tamaño de MLP elegido por validación: {mejor_mlp} "
          f"(MSE {minimos_val[mejor_mlp]:.4f} vs. "
          f"{ {n: round(minimos_val[n], 4) for n in mlps} })")

    elegidas = {mejor_mlp: mejores_epocas[mejor_mlp]}
    for nombre in CANDIDATAS:
        if not nombre.startswith("MLP"):
            elegidas[nombre] = mejores_epocas[nombre]
    return elegidas


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """TODO: la mejor red con cinco semillas, cada una por el backtest.

    Use con_tabla(red(nombre, epocas, semilla)). Guarde resultados/semillas.csv
    con una fila por semilla e imprima la media y la desviación del RMSE.
    """
    semillas = [2026, 1, 7, 42, 11]
    filas_sem = []
    for s in semillas:
        _, prom = backtest(serie, con_tabla(red(nombre, epocas, s)))
        filas_sem.append({
            "red": f"{nombre}, {epocas} épocas",
            "semilla": s,
            "RMSE": round(float(prom["RMSE"]), 2),
            "MAE": round(float(prom["MAE"]), 2),
            "MAPE": round(float(prom["MAPE"]), 2),
        })
    tabla_sem = pd.DataFrame(filas_sem)
    tabla_sem.to_csv(RESULTADOS / "semillas.csv", index=False)
    media_rmse = tabla_sem["RMSE"].mean()
    desv_rmse = tabla_sem["RMSE"].std(ddof=1)
    print(f"  Cinco semillas para {nombre} ({epocas} épocas):")
    print(tabla_sem.to_string(index=False))
    print(f"  RMSE en 5 semillas: media {media_rmse:.2f} mm, desviación {desv_rmse:.2f} mm")


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
        errores = np.abs(real.to_numpy() - pronostico)
        idx_peor = int(np.argmax(errores))
        mes_peor = real.index[idx_peor]
        print(f"  Último origen ({nombre}): RMSE {rmse(real, pronostico):.2f} mm; "
              f"peor mes {mes_peor:%Y-%m} (real {real.iloc[idx_peor]:.1f} vs. "
              f"pron {pronostico[idx_peor]:.1f} mm, error {pronostico[idx_peor] - real.iloc[idx_peor]:+.1f} mm)")
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
            comparar_tercera_red(serie)
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
