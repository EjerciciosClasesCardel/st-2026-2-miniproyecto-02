# Informe del miniproyecto 2

Llene cada sección reemplazando los marcadores `{{...}}`, sin quitar las
secciones. Las cifras salen de lo que imprime `python miniproyecto.py` y las
figuras son las que ese guion deja en `figuras/`. Las fórmulas van en LaTeX
(`$...$` o `$$...$$`).

## 1. La serie y el modelo a superar

La serie corresponde a la **precipitación mensual acumulada ($\text{mm/mes}$)** en la estación **CDEA El Topacio** (código `2629020601`), ubicada en la cuenca del río Pance (Cali, Valle del Cauca), suministrada por la **Corporación Autónoma Regional del Valle del Cauca (CVC)**. Abarca de **enero de 2015 a febrero de 2026 (`134` observaciones mensuales continuas)**, con un régimen bimodal andino (máximo estacional en abril con $+118.6\text{ mm}$ y mínimo en agosto con $-125.1\text{ mm}$) y un rango observado entre **$5.7\text{ mm}$** (`2019-08`) y **$577.0\text{ mm}$** (`2024-04`). Del Miniproyecto 1 se mantiene la misma función `preparar`: **no hay meses faltantes (`0` huecos)**, duplicados ni valores negativos, y **se conservan los `7` meses atípicos** detectados por el rango intercuartílico del residuo de la descomposición STL robusta ($[-174.5,\ 172.7]\text{ mm}$: `2025-12`, `2022-06`, `2024-04`, `2023-03`, `2022-05`, `2023-10` y `2022-02`), dejando señalados los meses con totales diarios a las 07:00 como posible artefacto de medición sin alterar la serie para no introducir fuga de información.

![La serie](../figuras/serie.png)

Del Miniproyecto 1 se traslada el mejor modelo de su tabla de métricas: **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`**, ajustado sobre la serie escalada $y_t / 100$ con diferenciación estacional ($d = 0,\ D = 1,\ m = 12$), un término de media móvil estacional ($Q = 1$) y el número de días del calendario del mes ($x_t = \text{dias\_mes}_t$, conocido con certeza de antemano sin fuga) como regresor exógeno:

$$(1 - B^{12})\bigl(y_t - \beta\,x_t\bigr) = (1 + \Theta_1 B^{12})\,\varepsilon_t, \qquad \text{es decir,} \qquad y_t = y_{t-12} + \beta\,(x_t - x_{t-12}) + \varepsilon_t + \Theta_1\,\varepsilon_{t-12},$$

donde en toda la serie $\Theta_1 = -0.853$, $\beta = -100.9\text{ mm/día}$ y se estiman **`3` parámetros** ($\beta$, $\Theta_1$ y $\sigma^2$). Al volver a pasar por el *backtest* común de `comun.py` (`13` orígenes desde `2024-02` hasta `2025-02`, horizonte $h = 1, \dots, 12$ sobre los últimos `24` meses `2024-03` a `2026-02`), sus métricas **coinciden exactamente con las del Miniproyecto 1**:

| Modelo clásico de referencia | RMSE (mm) | MAE (mm) | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`** | **113.79** | **91.75** | 54.31 | 3 | 3.5 |
| `SARIMA(0, 0, 0)(0, 1, 1, 12)` (sin exógena, ref. MP1) | 117.05 | 93.35 | 54.68 | 2 | 3.1 |
| `estacional ingenuo` (piso de referencia) | 154.93 | 112.39 | **43.90** | 0 | 0.0 |

No hubo ningún cambio en las métricas al trasladarlo (`RMSE = 113.79 mm`, `MAE = 91.75 mm`, `MAPE = 54.31 %`), y el guion imprime también la referencia del `SARIMA(0, 0, 0)(0, 1, 1, 12)` puro ($y_t - y_{t-12} = \varepsilon_t + \Theta_1 \varepsilon_{t-12}$, `RMSE = 117.05 mm`, `MAE = 93.35 mm`, `2` parámetros), mostrando que casi toda la reducción del error frente al `estacional ingenuo` (`-26.6 %` en RMSE y `-18.4 %` en MAE) proviene del término estacional $\Theta_1 \approx -0.853$, que amortigua los choques pluviales del año anterior hacia el promedio histórico de cada mes.

## 2. La tabla supervisada

Para transformar la serie univariada en un problema de aprendizaje supervisado $(X, z)$, elegimos **`OBJETIVO = "dif12"`**, de modo que cada modelo aprende la diferencia estacional con el mismo mes del año anterior:

$$z_t = \nabla_{12} y_t = y_t - y_{t-12}, \qquad \text{y reconstruye el nivel mediante} \qquad \hat{y}_t = y_{t-12} + \hat{z}_t.$$

Esta decisión mantiene coherencia con la prueba ADF del Miniproyecto 1 y del Taller 1 ($d = 0,\ D = 1$: la serie original con tendencia tiene $p = 0.2092$, mientras que $\nabla_{12} y_t$ es estacionaria con $p = 0.0033$), remueve de entrada el fuerte ciclo anual bimodal de la cuenca de Pance y permite que modelos basados en árboles o vecindades desplacen su rango dinámicamente respecto a $y_{t-12}$ sin quedar truncados por el techo histórico del entrenamiento cuando ocurre un récord pluvial como `2024-04` (`577.0 mm`).

**También probamos `OBJETIVO = "nivel"`** (con el mismo recorte en `0 mm`) y lo descartamos. Con `nivel` los árboles y KNN bajan su RMSE (`XGBoost` `133.45 mm`, `Random Forest` `135.03 mm`, `KNN` `141.18 mm`), porque promedian lluvias que ya vieron y nunca se alejan mucho de lo normal. Pero las redes empeoran (`LSTM` `154.80 mm` con `MAE = 134.88 mm`; `Conv1D` `154.32 mm`), y todos los modelos quedan con un MAPE de entre `66 %` y `93 %`, muy por encima del `43.90 %` del estacional ingenuo: en los meses secos pronostican lluvias de un mes normal. Con `dif12` la mejor red, la `LSTM`, se queda en `MAPE = 44.34 %`, y la elección del modelo final (el SARIMAX) no cambia con ninguno de los dos objetivos.

Para cada mes $t$, la función `tabla_supervisada(z)` construye **`14` columnas** que usan exclusivamente información disponible hasta $t - 1$:
- **`12` rezagos del objetivo (`rezago_1` a `rezago_12`):**

$$\text{rezago\_}k(t) = z_{t-k} = y_{t-k} - y_{t-k-12}, \qquad k = 1, 2, \dots, 12.$$

- **`1` variable de calendario (`mes`):** el mes del año extraído de la fecha del índice,

$$\text{mes}(t) = \text{month}(t) \in \{1, 2, \dots, 12\}.$$

- **`1` estadística de ventana móvil (`media_3`):** la media móvil trimestral calculada sobre la serie ya desplazada un mes (`z.shift(1).rolling(3).mean()`),

$$\text{media\_3}(t) = \frac{1}{3}\sum_{j=1}^{3} z_{t-j} = \frac{z_{t-1} + z_{t-2} + z_{t-3}}{3}.$$

De los `134` meses de la serie (`2015-01` a `2026-02`), se pierden `12` meses al calcular $z_t = y_t - y_{t-12}$ (todo `2015`) y otros `12` meses al exigir `rezago_12` completo (todo `2016`), quedando **`110` filas completas** desde `2017-01` hasta `2026-02`.

La función `comprobar_sin_fuga(serie)` verifica en el código la ausencia total de fuga de información (*data leakage*) en dos niveles: recorre cada uno de los `134` meses $i \in \{0, \dots, 133\}$, suma $+10\,000\text{ mm}$ tanto al objetivo $z_i$ como al nivel original $y_i$, recalcula la tabla supervisada completa y comprueba con `.iloc[i].equals(base.iloc[i])` que la fila del mes $i$ no sufre ningún cambio (si alguna cambiara, lanzaría `AssertionError`). Al correr `python miniproyecto.py`, el guion imprime:

```text
Tabla supervisada: 134 filas totales, 110 completas (2017-01 a 2026-02), 14 columnas: ['rezago_1', 'rezago_2', 'rezago_3', 'rezago_4', 'rezago_5', 'rezago_6', 'rezago_7', 'rezago_8', 'rezago_9', 'rezago_10', 'rezago_11', 'rezago_12', 'mes', 'media_3']
Sin fuga verificado: ninguna de las 134 filas cambia al alterar y(t) o z(t) en el mes que predice
```

El **pronóstico recursivo a 12 meses (`recursivo`)** y el **escalado por origen** operan dentro de `con_tabla`, cerrando las cuatro posibles fuentes de fuga de información:
1. **Sin fuga de escalado:** en cada uno de los `13` orígenes del *backtest*, `con_tabla` recibe únicamente la subserie `entrenamiento` hasta la fecha de corte, extrae sus filas completas $(X, z)$ y llama a `ajustado.fit(X, z)`. El escalador (`StandardScaler`) vive **dentro** de cada modelo (`make_pipeline(StandardScaler(), ...)` en `KNN` y `SVR`, y `self.escala_x = StandardScaler().fit(X)` junto con `self.media, self.desv = z.mean(), z.std()` dentro de `Red.fit`), por lo que se ajusta exclusivamente con las filas de entrenamiento de ese origen (de `86` filas en el primer origen `2024-02` a `98` filas en el último origen `2025-02`), sin ver jamás datos posteriores al corte.
2. **Sin fuga en rezagos ni ventanas (`shift(1)` y `padding=0`):** todas las columnas usan `z.shift(k)` con $k \ge 1$, la media móvil se calcula sobre `z.shift(1).rolling(3).mean()` y la `Conv1D` opera con `padding=0` (causal, sin rellenar con ceros al borde derecho ni centrar el filtro hacia $t + 1$).
3. **Sin barajar el tiempo ni elegir mirando el *backtest*:** los hiperparámetros y las épocas se eligen antes del primer origen (`serie.iloc[:-24]`) mediante `TimeSeriesSplit(n_splits=5, test_size=6)` y corte temporal de 12 meses (`X.iloc[:-12]` vs. `X.iloc[-12:]`), sin usar `KFold(shuffle=True)` ni tocar los 24 meses de prueba.
4. **Reconstrucción recursiva paso a paso:** para pronosticar los $h = 12$ meses siguientes ($s = 1, \dots, 12$), `recursivo` agrega a la copia de la serie el mes futuro $T + s$ con `np.nan`, recalcula `tabla_supervisada(objetivo(y)).iloc[[-1]]` para formar la fila de predictores de ese mes, obtiene $\hat{z}_{T+s}$ con `ajustado.predict`, lo devuelve al nivel sumando el mismo mes del año anterior ($\hat{y}_{T+s} = y_{T+s-12} + \hat{z}_{T+s}$) y escribe $\hat{y}_{T+s}$ en la serie como si fuera un dato observado para que alimente `rezago_1`, `rezago_2`, $\dots$ y `media_3` de los meses siguientes.

## 3. Aprendizaje automático

Los hiperparámetros de los cuatro modelos de aprendizaje automático se eligieron con `GridSearchCV` y validación temporal `TimeSeriesSplit(n_splits=5, test_size=6)` (criterio `neg_root_mean_squared_error`) usando únicamente `serie.iloc[:-24]`, es decir, las **`86` filas completas anteriores al primer origen del *backtest*** (`2017-01` a `2024-02`, donde los 5 pliegues entrenan con `56, 62, 68, 74` y `80` filas y validan bloques sucesivos de 6 meses entre `2021-09` y `2024-02`). Ninguno de los `24` meses del *backtest* (`2024-03` a `2026-02`) participa en la búsqueda: si los hiperparámetros se eligieran mirando los meses de prueba, el *backtest* dejaría de medir capacidad de generalización hacia el futuro y tendría fuga de selección.

| Modelo | Rejilla explorada | Hiperparámetros elegidos | RMSE de validación (mm) |
|---|---|---|---|
| **KNN** | `n_neighbors`: `[3, 5, 10, 15]` | `{'n_neighbors': 5}` | 135.97 |
| **SVR (RBF)** | `C`: `[0.1, 1.0, 10.0, 100.0]`, `epsilon`: `[0.01, 0.1, 0.5]` | `{'C': 100.0, 'epsilon': 0.5}` | 135.18 |
| **Random Forest** | `max_depth`: `[3, 5, 10, None]`, `min_samples_leaf`: `[1, 3, 5]` | `{'max_depth': 10, 'min_samples_leaf': 5}` | 132.53 |
| **XGBoost** | `max_depth`: `[2, 3, 5]`, `learning_rate`: `[0.01, 0.05, 0.1]` | `{'max_depth': 2, 'learning_rate': 0.01}` | **131.95** |

![Importancia de las variables](../figuras/importancia.png)

Al ajustar ambos ensambles sobre las `86` filas previas con sus hiperparámetros elegidos, la importancia de las variables (`feature_importances_`) muestra un patrón consistente con el análisis clásico del Miniproyecto 1:
- **En Random Forest:** domina ampliamente **`rezago_12` con `0.4315` (`43.15 %` de la importancia total)**, casi cuatro veces más que la segunda variable (`rezago_2 = 0.1168`, `rezago_8 = 0.0910`, `rezago_10 = 0.0514`, `rezago_7 = 0.0476`, `rezago_1 = 0.0435`, `media_3 = 0.0420`), mientras que **`mes` ocupa el último lugar con apenas `0.0151` (`1.51 %`)**.
- **En XGBoost:** también lidera **`rezago_12` con `0.2102` (`21.02 %`)**, seguido de `rezago_2` (`0.1503`), `rezago_4` (`0.1209`), `rezago_8` (`0.1061`), `rezago_11` (`0.0908`), `rezago_10` (`0.0850`) y `rezago_7` (`0.0814`), mientras que **`mes` y `media_3` reciben importancia `0.0000` (`0.00 %`)**.
- **Conexión con la ACF y PACF del Miniproyecto 1:** en el Miniproyecto 1, los únicos rezagos dentro del primer año que salían de la banda de confianza ($\pm 0.177$) en la ACF y la PACF de $z_t = y_t - y_{t-12}$ eran el **rezago `12`** ($\text{ACF}_{12} = -0.35$, $\text{PACF}_{12} = -0.41$) y el **rezago `7`** ($\text{ACF}_7 = +0.27$, $\text{PACF}_7 = +0.27$). Ambos ensambles descubren por sí solos que `rezago_12` es el predictor fundamental (por la reversión a la media interanual inducida al compartir $y_{t-12}$ con signo opuesto en $z_t = y_t - y_{t-12}$ y $z_{t-12} = y_{t-12} - y_{t-24}$) e incluyen también a `rezago_7` en su grupo superior (`4.76 %` y `8.14 %`). El hecho de que `mes` pese casi cero (`1.51 %` y `0.00 %`) confirma que la diferencia estacional `dif12` ya eliminó el ciclo anual determinístico.

En el *backtest* común sobre los últimos 24 meses, los cuatro modelos de aprendizaje automático obtienen:

| Modelo de aprendizaje automático | RMSE (mm) | MAE (mm) | MAPE (%) | Parámetros / nodos | Segundos |
|---|---|---|---|---|---|
| **`SVR` (`C=100, epsilon=0.5`)** | **150.93** | **113.94** | 55.25 | 1 471 | 0.7 |
| `XGBoost` (`max_depth=2, lr=0.01`) | 154.27 | 117.69 | **51.23** | 1 374 | 3.4 |
| `Random Forest` (`max_depth=10, leaf=5`) | 158.36 | 121.56 | 58.72 | 1 904 | 3.4 |
| `KNN` (`n_neighbors=5`) | 165.65 | 132.21 | 69.75 | 1 372 | 0.7 |

**Ganador entre los cuatro de aprendizaje automático:** **`SVR`** con **`RMSE = 150.93 mm`** y **`MAE = 113.94 mm`**, seguido de **`XGBoost`** (`RMSE = 154.27 mm`, que mejora en `10.06 mm` al XGBoost sin regularizar del Taller 3 gracias a limitar `max_depth = 2` y `learning_rate = 0.01`). Tanto `SVR` como `XGBoost` superan en RMSE al `estacional ingenuo` (`154.93 mm`), mientras que `Random Forest` (`158.36 mm`) y sobre todo `KNN` (`165.65 mm`, arrastrado por vecinos post-pico con diferencias muy negativas) quedan por encima del piso ingenuo.

## 4. Aprendizaje profundo

Evaluamos tres arquitecturas en PyTorch entrenadas mediante la clase `Red` con optimizador Adam (`lr = 0.01`), función de pérdida MSE sobre objetivo y entradas estandarizadas por origen, y lote completo hasta `300` épocas:

1. **Perceptrón Multicapa (`MLP`, `secuencia=False`):** recibe las `14` columnas de la tabla supervisada como un vector plano $(n, 14)$ estandarizado con `StandardScaler`, aplica una capa oculta densa con activación ReLU y una capa lineal de salida:

$$\hat{z}^{(\text{esc})} = W_2\,\text{ReLU}(W_1 x^{(\text{esc})} + b_1) + b_2.$$

   Se compararon dos tamaños de capa oculta: **`MLP (8 ocultas)`** ($14 \times 8 + 8 + 8 \times 1 + 1 = \mathbf{129}$ **parámetros**) y **`MLP (32 ocultas)`** ($14 \times 32 + 32 + 32 \times 1 + 1 = \mathbf{513}$ **parámetros**).
2. **Red Recurrente `LSTM (16 ocultas)` (`secuencia=True`):** recibe únicamente los `12` rezagos ordenados cronológicamente del más antiguo (`rezago_12`) al más reciente (`rezago_1`) como un tensor tridimensional de forma $(n, 12, 1)$ estandarizado con la media y desviación de $z$, procesa un mes por paso a través de una celda `nn.LSTM(1, 16, batch_first=True)` con compuertas de olvido, entrada y salida, y proyecta el último estado oculto $h_{12} \in \mathbb{R}^{16}$ con `nn.Linear(16, 1)`. Cuenta con $4 \times (1 \times 16 + 16 \times 16 + 16 + 16) + (16 \times 1 + 1) = \mathbf{1\,233}$ **parámetros**.
3. **Tercera red a elección — `Conv1D (8 canales)` (`secuencia=True`):** recibe la secuencia ordenada $(n, 12, 1)$ y la transpone con `x.transpose(1, 2)` a $(n, 1, 12)$ (porque `nn.Conv1d` espera canales en el eje 1 y tiempo en el eje 2; sin transponer trataría los 12 meses como 12 canales de largo 1), aplica `nn.Conv1d(1, 8, kernel_size=3, padding=0)` (sin relleno para no mezclar meses reales con ceros inventados al borde derecho ni mirar hacia el futuro) seguida de `ReLU`, resume las $12 - 3 + 1 = 10$ posiciones mediante promedio temporal adaptativo `nn.AdaptiveAvgPool1d(1)` y predice con `nn.Linear(8, 1)`.
   - **Por qué elegimos `Conv1D + AdaptiveAvgPool1d` (`41` parámetros) frente a las otras alternativas** (verificadas en `comparar_tercera_red` de `miniproyecto.py`):
     1. *Frente a `Conv1D + flatten(1)` (`113` parámetros):* al aplanar el mapa $(n, 8, 10)$ a `80` valores y pasar por `Linear(80, 1)`, cada posición del año recibe su propio peso como en un MLP y pierde la invarianza a traslación temporal, subiendo el RMSE en Pance de **`152.49 mm` (`MAPE = 43.11 %`)** a **`175.03 mm` (`MAPE = 106.47 %`, `34` épocas)**. Con `AdaptiveAvgPool1d(1)`, la red baja a $8 \times (3 + 1) + (8 + 1) = \mathbf{41}$ **parámetros** (30 veces menos que la LSTM) y logra el menor MAPE de toda la tabla (`43.11 %`).
     2. *Frente al `Híbrido SARIMAX + MLP(16) sobre sus residuos` (`3 + 241 = 244` parámetros):* un híbrido solo mejora al modelo clásico cuando el residuo del SARIMA aún conserva autocorrelación ($p < 0.05$ en Ljung-Box). En nuestra serie, al quitar los primeros $d + 12D = 12$ meses de arranque, la prueba de Ljung-Box sobre el residuo del `SARIMAX(0, 0, 0)(0, 1, 1, 12)` da **$p(12) = 0.4689$ y $p(24) = 0.6623$** ($p > 0.05$: ruido blanco). Por eso, al entrenar un `MLP(16)` sobre ese residuo (`6` épocas), el híbrido obtiene **`RMSE = 116.23 mm` y `MAE = 93.74 mm`**, empeorando en `+2.44 mm` al `SARIMAX` solo (`113.79 mm`) por ajustar ruido.
     3. *Frente al `Transformer` de 1 capa (`d_model=16, nhead=2, dim_feedforward=32`):* aun sumando codificación sinusoidal fija de posición via `register_buffer` (`192` valores fijos, `0` parámetros entrenables, necesaria para que la autoatención distinga el orden temporal) y `batch_first=True`, la capa `TransformerEncoderLayer` tiene **`2 273` parámetros entrenables** (`32` de proyección de entrada, `1 088` en $Q, K, V$ y salida de atención, `1 072` en la red *feed-forward*, `64` en *LayerNorm* y `17` de salida), es decir, **`30.7` parámetros por ventana de entrenamiento** (`74` filas). Con tan pocas ventanas los pesos de autoatención se aplanan hacia el promedio uniforme ($1/12 \approx 0.083$), obteniendo en el *backtest* **`RMSE = 152.00 mm`, `MAE = 109.35 mm`, `MAPE = 43.35 %` (`12` épocas)**, prácticamente idéntico a la `Conv1D` (`RMSE = 152.49 mm`, `MAPE = 43.11 %`) pero con **`55` veces más parámetros**.

![Curvas de pérdida](../figuras/perdida.png)

La validación temporal de las redes se realizó sobre las `86` filas anteriores al *backtest*, entrenando con las primeras **`74` filas** (`2017-01` a `2023-02`) y validando con los últimos **`12` meses** (`2023-03` a `2024-02`):

| Red candidata | Parámetros | Mejor época | MSE val. mínimo | MSE ent. en mejor época | MSE val. época 300 | MSE ent. época 300 |
|---|---|---|---|---|---|---|
| `MLP (8 ocultas)` | 129 | 14 | 1.5716 | 0.7211 | 9.2633 | 0.0005 |
| **`MLP (32 ocultas)` (MLP elegido)** | **513** | **11** | **1.4099** | **0.6292** | 2.7354 | 0.0000 |
| **`LSTM (16 ocultas)`** | **1 233** | **37** | **1.4656** | **0.7834** | 3.5598 | 0.0002 |
| **`Conv1D (8 canales)`** | **41** | **21** | **1.7224** | **0.9585** | 2.2955 | 0.8084 |

**Lectura de las curvas de pérdida:**
- Con `TASA = 0.01` y solo `74` filas de entrenamiento, **el sobreajuste empieza muy temprano en todas las redes**: en los dos MLP la pérdida de validación toca fondo entre las épocas **`11` y `14`** (`MSE = 1.4099` en `MLP 32` y `1.5716` en `MLP 8`), y a partir de ahí la pérdida de entrenamiento cae hasta **`0.0000`–`0.0005`** (memorización completa de las 74 filas) mientras la validación se dispara hasta `2.74` y `9.26`.
- En la **`LSTM (16 ocultas)`** el descenso inicial es más gradual: alcanza su mínimo de validación en la **época `37`** (`MSE = 1.4656`, con entrenamiento `0.7834`), antes de sobreajustar entre las épocas `40` y `100` (donde el entrenamiento cae a `0.0002` y la validación sube a `3.56`).
- En la **`Conv1D (8 canales)`**, sus `41` parámetros impiden que memorice el entrenamiento (en la época 300 el MSE de entrenamiento sigue en `0.8084`), alcanzando su mínimo de validación en la **época `21`** (`MSE = 1.7224`).
- Entre los dos tamaños de MLP, la validación elige **`MLP (32 ocultas)` con `11` épocas** (`MSE = 1.4099` frente a `1.5716` de `8` ocultas), que pasa al *backtest* junto con **`LSTM (16 ocultas), 37 épocas`** y **`Conv1D (8 canales), 21 épocas`**.

En el *backtest*, la mejor de las tres redes es **`LSTM (16 ocultas), 37 épocas`** (`RMSE = 137.37 mm`, `MAE = 99.40 mm`, `MAPE = 44.34 %`). Al repetir su *backtest* con **cinco semillas** (`resultados/semillas.csv`) se obtiene:

| Red | Semilla | RMSE (mm) | MAE (mm) | MAPE (%) |
|---|---|---|---|---|
| `LSTM (16 ocultas), 37 épocas` | `2026` | 137.37 | **99.40** | **44.34** |
| `LSTM (16 ocultas), 37 épocas` | `1` | 157.59 | 122.97 | 60.16 |
| `LSTM (16 ocultas), 37 épocas` | `7` | 152.96 | 116.18 | 56.38 |
| `LSTM (16 ocultas), 37 épocas` | `42` | 138.58 | 105.35 | 49.58 |
| `LSTM (16 ocultas), 37 épocas` | `11` | **132.92** | 103.36 | 52.59 |
| **Media de las 5 semillas** | — | **143.88** | **109.45** | **52.61** |
| **Desviación estándar muestral ($s$)** | — | **10.74** | **9.78** | **6.10** |

El RMSE de la LSTM varía entre `132.92 mm` (semilla `11`) y `157.59 mm` (semilla `1`), con una **media de `143.88 mm` y una desviación estándar de `10.74 mm`**. La ventaja del modelo clásico `SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` (`RMSE = 113.79 mm`) es de **`23.58 mm` frente a la semilla `2026`** ($2.20$ desviaciones estándar) y de **`30.09 mm` frente a la media de las cinco semillas** ($2.80$ desviaciones estándar; incluso frente a la mejor semilla `11`, el clásico gana por `19.13 mm`, $1.78$ desviaciones estándar). Por tanto, **la diferencia a favor del modelo clásico es significativamente mayor que la desviación entre semillas**: la derrota de la red no se debe a mala suerte en la inicialización aleatoria de los pesos.

## 5. La tabla y la elección

La tabla completa generada por `python miniproyecto.py` en `resultados/metricas.csv`, ordenada de menor a mayor RMSE y sin modificar `comun.py`, es:

| Modelo | RMSE | MAE | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`** | **113.79** | **91.75** | 54.31 | **3** | 3.3 |
| `LSTM (16 ocultas), 37 épocas` | 137.37 | 99.40 | 44.34 | 1 233 | 2.3 |
| `SVR` | 150.93 | 113.94 | 55.25 | 1 471 | **0.7** |
| `Conv1D (8 canales), 21 épocas` | 152.49 | 109.64 | **43.11** | 41 | 1.8 |
| `MLP (32 ocultas), 11 épocas` | 154.03 | 113.47 | 66.20 | 513 | 1.2 |
| `XGBoost` | 154.27 | 117.69 | 51.23 | 1 374 | 3.4 |
| `estacional ingenuo` | 154.93 | 112.39 | 43.90 | 0 | 0.0 |
| `Random Forest` | 158.36 | 121.56 | 58.72 | 1 904 | 3.4 |
| `KNN` | 165.65 | 132.21 | 69.75 | 1 372 | **0.7** |

![Pronósticos desde el último origen](../figuras/pronosticos.png)

**¿Por qué ninguna red neuronal ni modelo de aprendizaje automático supera al clásico en nuestra serie?** Lo explican cuatro datos concretos de la serie de Pance:

1. **Desproporción entre número de parámetros y meses disponibles:** en los `13` orígenes del *backtest*, la tabla supervisada solo dispone de **`86` a `98` filas completas** de entrenamiento (`74` filas al descontar validación). Mientras el `SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` estima **`3` parámetros** (`2` en el SARIMA puro, a razón de $\approx 49$ observaciones por parámetro), el `MLP (32 ocultas)` ajusta **`513` pesos** ($5.2$ parámetros por fila) y la `LSTM (16 ocultas)` ajusta **`1 233` pesos** ($12.6$ parámetros por fila). Como muestran las curvas de pérdida (`figuras/perdida.png`), a partir de la época `11` en el MLP y de la `37` en la LSTM las redes memorizan el ruido meteorológico de entrenamiento (`MSE` $\to 0.0000$).
2. **Estructura parsimoniosa de la dependencia temporal:** como demostraron la ACF/PACF del Miniproyecto 1 y las importancias de variables (`figuras/importancia.png`, donde `rezago_12` concentra `43.15 %` en Random Forest y `21.02 %` en XGBoost), la única señal lineal fuerte en $z_t = y_t - y_{t-12}$ es la autocorrelación negativa en el rezago 12 ($\rho_{12} = -0.35$) producida por el rebote hacia la media tras un año anómalo. El parámetro $\Theta_1 = -0.853$ del SARIMA captura exactamente esa relación con una sola ecuación en diferencias, mientras que las redes y árboles intentan combinar `12` a `14` entradas ruidosas y propagan esos errores a lo largo de los `12` pasos recursivos.
3. **El mes crítico en que fallan los modelos y el problema de los récords climáticos (`2026-01` en el último origen y `2024-04 = 577.0 mm` en los primeros orígenes):** las redes neuronales entrenadas con pérdida cuadrática media (MSE) tienden a suavizar y fallar sistemáticamente en los récords o rebotes climáticos extremos que salen de la distribución histórica de entrenamiento (Zhang et al., 2026). En el pronóstico a 12 meses desde el último origen (`2025-02` $\to$ `2025-03` a `2026-02`, `figuras/pronosticos.png`), el peor error de los tres modelos ocurre en **enero de 2026 (`2026-01`, lluvia real = `315.2 mm`)**:
   - En enero de 2024 hubo una sequía extrema (`15.0 mm`) y enero de 2025 siguió siendo inusualmente seco (`53.8 mm`), mientras que febrero de 2025 tuvo un salto húmedo (`464.0 mm`, $z_{\text{2025-02}} = +313.6\text{ mm}$).
   - Al reconstruir el nivel como $\hat{y}_{\text{2026-01}} = y_{\text{2025-01}} + \hat{z}_{\text{2026-01}} = 53.8 + \hat{z}_{\text{2026-01}}$, tanto **`SVR`** como **`LSTM (16 ocultas)`** arrastran diferencias recursivas negativas tras el pico de febrero y llegan a lluvias físicamente imposibles (`-48.9 mm` en `SVR` y `-12.3 mm` en `LSTM`). Por eso `a_nivel` recorta todo pronóstico en `0 mm`: con el recorte ambos pronostican **`0.0 mm` (error `-315.2 mm`)**, con RMSE del último origen de **`153.87 mm` en `SVR`** y **`150.83 mm` en `LSTM`** (sin recorte eran `162.61 mm` y `152.99 mm`). El recorte nunca empeora un pronóstico y mejora un poco a todos los modelos con `dif12` (por ejemplo, la `LSTM` baja de `137.84` a `137.37 mm` y `KNN` de `168.96` a `165.65 mm`), pero no cambia la conclusión: el modelo sigue sin ver venir el enero lluvioso.
   - En cambio, el **`SARIMAX`** ($\Theta_1 = -0.853$) descuenta el $85.3\,\%$ de la sequía anormal de `2025-01` y pronostica **`135.6 mm` en `2026-01` (error `-179.6 mm`, RMSE del último origen `102.48 mm`)**, evitando el colapso.
4. **Inestabilidad entre semillas:** en la `LSTM (16 ocultas)` el simple cambio de semilla hace variar el RMSE en un rango de **`24.67 mm`** (de `132.92 mm` con semilla `11` a `157.59 mm` con semilla `1`, desviación de `10.74 mm`), mientras que el modelo clásico es determinístico dado el máximo verosímil.

## 6. Conclusión

El grupo **pondría en uso el modelo clásico del Miniproyecto 1 —`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` (o su versión `SARIMA(0, 0, 0)(0, 1, 1, 12)` pura de `2` parámetros, dado que la exógena de días del mes solo modifica febreros bisiestos)**— y, como alternativa de aprendizaje profundo únicamente si se exigiera una arquitectura neuronal, la **`LSTM (16 ocultas), 37 épocas`** (o la **`Conv1D (8 canales), 21 épocas`** por su bajo MAPE de `43.11 %` con solo `41` parámetros).

Frente al modelo clásico (`RMSE = 113.79 mm`, `MAE = 91.75 mm`, `3` parámetros, `3.3 s`), elegir la mejor red neuronal (**`LSTM (16 ocultas), 37 épocas`**, `RMSE = 137.37 mm`, `MAE = 99.40 mm`) implica pagar un costo desfavorable en las tres dimensiones operativas y un mayor error (`+23.58 mm` en RMSE y `+7.65 mm` en MAE):
- **En datos:** el SARIMA/SARIMAX opera desde el mes `13` (solo pierde `12` meses por la diferencia estacional $D = 1$, aprovechando `122` observaciones efectivas), mientras que la tabla supervisada y las ventanas de la LSTM con `OBJETIVO = "dif12"` y `12` rezagos **pierden `24` meses de historia** (todo `2015` y `2016`, dejando solo `86` a `98` filas por origen), un sacrificio crítico en series hidrológicas mensuales cortas.
- **En tiempo de entrenamiento y estabilidad:** aunque un *backtest* de la LSTM con `37` épocas fijas en CPU tarda **`2.3 s`** (similar a los **`3.3 s`** del SARIMAX y a los **`3.4 s`** de Random Forest), elegir sus épocas con validación previa y estabilizar su varianza entre semillas ($\pm 10.74\text{ mm}$ de RMSE en `5` semillas) multiplica por `5` el tiempo de cómputo ($\approx 14\text{ s}$).
- **En explicabilidad:** el modelo clásico se resume en un único parámetro interpretable ($\Theta_1 = -0.853$: cada mes se pronostica como el promedio histórico de ese mes más el $14.7\,\%$ de la anomalía del año anterior) y garantiza niveles coherentes con la climatología de la cuenca de Pance, mientras que la LSTM distribuye su decisión en **`1 233` parámetros opacos** de compuertas recurrentes que, ante secuencias atípicas en modo recursivo, pueden predecir precipitaciones negativas (`-12.3 mm` en `2026-01` antes del recorte en `0 mm`).
- **Modelo local frente a modelo global — ¿cambiaría la elección si tuviéramos que pronosticar mil series como la nuestra?:** **sí cambiaría.** Para **una sola serie mensual corta** (la estación CDEA El Topacio aislada, con `86`–`98` filas supervisadas), un **modelo local** parsimonioso como `SARIMAX` (`3` parámetros) es imbatible porque los modelos con cientos o miles de parámetros sobreajustan. Sin embargo, si la CVC tuviera que pronosticar simultáneamente **`1 000` estaciones pluviométricas** de la red hidroclimatológica andina, apilar las `1 000` series estandarizadas en una única tabla supervisada con covariables de cuenca y altitud (**modelo global**, Montero-Manso y Hyndman, 2021) proporcionaría **`~110 000` ventanas de entrenamiento**. En ese régimen ($< 0.02$ parámetros por ventana), un **`XGBoost`** o una red global (**`LSTM` / `Transformer`**) aprendería patrones regionales compartidos (como anomalías de El Niño y La Niña entre cuencas vecinas) sin sobreajustar y evitaría ajustar, diagnosticar y mantener `1 000` modelos SARIMA individuales.
