# Informe del miniproyecto 2

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

El **pronóstico recursivo a 12 meses (`recursivo`)** y el **escalado por origen** operan dentro de `con_tabla`:
1. En cada uno de los `13` orígenes del *backtest*, `con_tabla` recibe únicamente la subserie `entrenamiento` hasta la fecha de corte, extrae sus filas completas $(X, z)$ y llama a `ajustado.fit(X, z)`.
2. El escalador (`StandardScaler`) vive **dentro** de cada modelo (`make_pipeline(StandardScaler(), ...)` en `KNN` y `SVR`, y `self.escala_x = StandardScaler().fit(X)` junto con `self.media, self.desv = z.mean(), z.std()` dentro de `Red.fit`), por lo que se ajusta exclusivamente con las filas de entrenamiento de ese origen (de `86` filas en el primer origen `2024-02` a `98` filas en el último origen `2025-02`), sin ver jamás datos posteriores al corte.
3. Para pronosticar los $h = 12$ meses siguientes ($s = 1, \dots, 12$), `recursivo` agrega a la copia de la serie el mes futuro $T + s$ con `np.nan`, recalcula `tabla_supervisada(objetivo(y)).iloc[[-1]]` para formar la fila de predictores de ese mes, obtiene $\hat{z}_{T+s}$ con `ajustado.predict`, lo devuelve al nivel sumando el mismo mes del año anterior ($\hat{y}_{T+s} = y_{T+s-12} + \hat{z}_{T+s}$) y escribe $\hat{y}_{T+s}$ en la serie como si fuera un dato observado para que alimente `rezago_1`, `rezago_2`, $\dots$ y `media_3` de los meses siguientes.

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
| **`SVR` (`C=100, epsilon=0.5`)** | **151.61** | **114.24** | 55.34 | 1 471 | 1.1 |
| `XGBoost` (`max_depth=2, lr=0.01`) | 154.27 | 117.69 | **51.23** | 1 374 | 1.9 |
| `Random Forest` (`max_depth=10, leaf=5`) | 158.44 | 121.61 | 58.74 | 1 904 | 6.2 |
| `KNN` (`n_neighbors=5`) | 168.96 | 134.03 | 70.75 | 1 372 | 1.7 |

**Ganador entre los cuatro de aprendizaje automático:** **`SVR`** con **`RMSE = 151.61 mm`** y **`MAE = 114.24 mm`**, seguido de **`XGBoost`** (`RMSE = 154.27 mm`, que mejora en `10.06 mm` al XGBoost sin regularizar del Taller 3 gracias a limitar `max_depth = 2` y `learning_rate = 0.01`). Tanto `SVR` como `XGBoost` superan en RMSE al `estacional ingenuo` (`154.93 mm`), mientras que `Random Forest` (`158.44 mm`) y sobre todo `KNN` (`168.96 mm`, arrastrado por vecinos post-pico con diferencias muy negativas) quedan por encima del piso ingenuo.

## 4. Aprendizaje profundo

Evaluamos tres arquitecturas en PyTorch entrenadas mediante la clase `Red` con optimizador Adam (`lr = 0.01`), función de pérdida MSE sobre objetivo y entradas estandarizadas por origen, y lote completo hasta `300` épocas:

1. **Perceptrón Multicapa (`MLP`, `secuencia=False`):** recibe las `14` columnas de la tabla supervisada como un vector plano $(n, 14)$ estandarizado con `StandardScaler`, aplica una capa oculta densa con activación ReLU y una capa lineal de salida:

$$\hat{z}^{(\text{esc})} = W_2\,\text{ReLU}(W_1 x^{(\text{esc})} + b_1) + b_2.$$

   Se compararon dos tamaños de capa oculta: **`MLP (8 ocultas)`** ($14 \times 8 + 8 + 8 \times 1 + 1 = \mathbf{129}$ **parámetros**) y **`MLP (32 ocultas)`** ($14 \times 32 + 32 + 32 \times 1 + 1 = \mathbf{513}$ **parámetros**).
2. **Red Recurrente `LSTM (16 ocultas)` (`secuencia=True`):** recibe únicamente los `12` rezagos ordenados cronológicamente del más antiguo (`rezago_12`) al más reciente (`rezago_1`) como un tensor tridimensional de forma $(n, 12, 1)$ estandarizado con la media y desviación de $z$, procesa un mes por paso a través de una celda `nn.LSTM(1, 16, batch_first=True)` con compuertas de olvido, entrada y salida, y proyecta el último estado oculto $h_{12} \in \mathbb{R}^{16}$ con `nn.Linear(16, 1)`. Cuenta con $4 \times (1 \times 16 + 16 \times 16 + 16 + 16) + (16 \times 1 + 1) = \mathbf{1\,233}$ **parámetros**.
3. **Tercera red a elección — `Conv1D (8 canales)` (`secuencia=True`):** recibe la misma secuencia ordenada $(n, 12, 1)$ transpuesta a $(n, 1, 12)$, aplica una convolución temporal `nn.Conv1d(1, 8, kernel_size=3)` seguida de `ReLU`, resume los `10` pasos resultantes mediante promedio temporal adaptativo `nn.AdaptiveAvgPool1d(1)` y predice con `nn.Linear(8, 1)`. **Por qué la elegimos:** con tan solo $8 \times 1 \times 3 + 8 + 8 \times 1 + 1 = \mathbf{41}$ **parámetros** (30 veces menos que la LSTM y 12 veces menos que el MLP de 32 neuronas), sus filtros locales de 3 meses detectan patrones trimestrales de cambio pluvial invariantes a la posición dentro del año sin memorizar las `74`–`98` filas de entrenamiento.

![Curvas de pérdida](../figuras/perdida.png)

La validación temporal de las redes se realizó sobre las `86` filas anteriores al *backtest*, entrenando con las primeras **`74` filas** (`2017-01` a `2023-02`) y validando con los últimos **`12` meses** (`2023-03` a `2024-02`):

| Red candidata | Parámetros | Mejor época | MSE val. mínimo | MSE ent. en mejor época | MSE val. época 300 | MSE ent. época 300 |
|---|---|---|---|---|---|---|
| `MLP (8 ocultas)` | 129 | 14 | 1.5716 | 0.7211 | 9.2633 | 0.0005 |
| **`MLP (32 ocultas)` (MLP elegido)** | **513** | **11** | **1.4099** | **0.6292** | 2.7354 | 0.0000 |
| **`LSTM (16 ocultas)`** | **1 233** | **37** | **1.4656** | **0.7834** | 3.5591 | 0.0002 |
| **`Conv1D (8 canales)`** | **41** | **21** | **1.7224** | **0.9585** | 2.2955 | 0.8084 |

**Lectura de las curvas de pérdida:**
- Con `TASA = 0.01` y solo `74` filas de entrenamiento, **el sobreajuste empieza muy temprano en todas las redes**: en los dos MLP la pérdida de validación toca fondo entre las épocas **`11` y `14`** (`MSE = 1.4099` en `MLP 32` y `1.5716` en `MLP 8`), y a partir de ahí la pérdida de entrenamiento cae hasta **`0.0000`–`0.0005`** (memorización completa de las 74 filas) mientras la validación se dispara hasta `2.74` y `9.26`.
- En la **`LSTM (16 ocultas)`** el descenso inicial es más gradual: alcanza su mínimo de validación en la **época `37`** (`MSE = 1.4656`, con entrenamiento `0.7834`), antes de sobreajustar entre las épocas `40` y `100` (donde el entrenamiento cae a `0.0002` y la validación sube a `3.56`).
- En la **`Conv1D (8 canales)`**, sus `41` parámetros impiden que memorice el entrenamiento (en la época 300 el MSE de entrenamiento sigue en `0.8084`), alcanzando su mínimo de validación en la **época `21`** (`MSE = 1.7224`).
- Entre los dos tamaños de MLP, la validación elige **`MLP (32 ocultas)` con `11` épocas** (`MSE = 1.4099` frente a `1.5716` de `8` ocultas), que pasa al *backtest* junto con **`LSTM (16 ocultas), 37 épocas`** y **`Conv1D (8 canales), 21 épocas`**.

En el *backtest*, la mejor de las tres redes es **`LSTM (16 ocultas), 37 épocas`** (`RMSE = 137.84 mm`, `MAE = 99.89 mm`, `MAPE = 44.89 %`). Al repetir su *backtest* con **cinco semillas** (`resultados/semillas.csv`) se obtiene:

| Red | Semilla | RMSE (mm) | MAE (mm) | MAPE (%) |
|---|---|---|---|---|
| `LSTM (16 ocultas), 37 épocas` | `2026` | 137.84 | **99.89** | **44.89** |
| `LSTM (16 ocultas), 37 épocas` | `1` | 159.74 | 124.29 | 61.05 |
| `LSTM (16 ocultas), 37 épocas` | `7` | 154.19 | 117.15 | 57.18 |
| `LSTM (16 ocultas), 37 épocas` | `42` | 139.55 | 106.48 | 51.04 |
| `LSTM (16 ocultas), 37 épocas` | `11` | **133.66** | 104.35 | 53.67 |
| **Media de las 5 semillas** | — | **145.00** | **110.43** | **53.57** |
| **Desviación estándar muestral ($s$)** | — | **11.31** | **9.99** | **6.13** |

El RMSE de la LSTM varía entre `133.66 mm` (semilla `11`) y `159.74 mm` (semilla `1`), con una **media de `145.00 mm` y una desviación estándar de `11.31 mm`**. La ventaja del modelo clásico `SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` (`RMSE = 113.79 mm`) es de **`24.05 mm` frente a la semilla `2026`** ($2.13$ desviaciones estándar) y de **`31.21 mm` frente a la media de las cinco semillas** ($2.76$ desviaciones estándar; incluso frente a la mejor semilla `11`, el clásico gana por `19.87 mm`, $1.76$ desviaciones estándar). Por tanto, **la diferencia a favor del modelo clásico es significativamente mayor que la desviación entre semillas**: la derrota de la red no se debe a mala suerte en la inicialización aleatoria de los pesos.

## 5. La tabla y la elección

La tabla completa generada por `python miniproyecto.py` en `resultados/metricas.csv`, ordenada de menor a mayor RMSE y sin modificar `comun.py`, es:

| Modelo | RMSE | MAE | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`** | **113.79** | **91.75** | 54.31 | **3** | 3.3 |
| `LSTM (16 ocultas), 37 épocas` | 137.84 | 99.89 | 44.89 | 1 233 | 2.7 |
| `SVR` | 151.61 | 114.24 | 55.34 | 1 471 | **1.1** |
| `Conv1D (8 canales), 21 épocas` | 152.49 | 109.64 | **43.11** | 41 | 1.4 |
| `XGBoost` | 154.27 | 117.69 | 51.23 | 1 374 | 2.1 |
| `MLP (32 ocultas), 11 épocas` | 154.50 | 113.74 | 66.28 | 513 | 1.3 |
| `estacional ingenuo` | 154.93 | 112.39 | 43.90 | 0 | 0.0 |
| `Random Forest` | 158.44 | 121.61 | 58.74 | 1 904 | 4.6 |
| `KNN` | 168.96 | 134.03 | 70.75 | 1 372 | 1.2 |

![Pronósticos desde el último origen](../figuras/pronosticos.png)

**¿Por qué ninguna red neuronal ni modelo de aprendizaje automático supera al clásico en nuestra serie?** Lo explican cuatro datos concretos de la serie de Pance:

1. **Desproporción entre número de parámetros y meses disponibles:** en los `13` orígenes del *backtest*, la tabla supervisada solo dispone de **`86` a `98` filas completas** de entrenamiento (`74` filas al descontar validación). Mientras el `SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` estima **`3` parámetros** (`2` en el SARIMA puro, a razón de $\approx 49$ observaciones por parámetro), el `MLP (32 ocultas)` ajusta **`513` pesos** ($5.2$ parámetros por fila) y la `LSTM (16 ocultas)` ajusta **`1 233` pesos** ($12.6$ parámetros por fila). Como muestran las curvas de pérdida (`figuras/perdida.png`), a partir de la época `11` en el MLP y de la `37` en la LSTM las redes memorizan el ruido meteorológico de entrenamiento (`MSE` $\to 0.0000$).
2. **Estructura parsimoniosa de la dependencia temporal:** como demostraron la ACF/PACF del Miniproyecto 1 y las importancias de variables (`figuras/importancia.png`, donde `rezago_12` concentra `43.15 %` en Random Forest y `21.02 %` en XGBoost), la única señal lineal fuerte en $z_t = y_t - y_{t-12}$ es la autocorrelación negativa en el rezago 12 ($\rho_{12} = -0.35$) producida por el rebote hacia la media tras un año anómalo. El parámetro $\Theta_1 = -0.853$ del SARIMA captura exactamente esa relación con una sola ecuación en diferencias, mientras que las redes y árboles intentan combinar `12` a `14` entradas ruidosas y propagan esos errores a lo largo de los `12` pasos recursivos.
3. **El mes crítico en que fallan los modelos (`2026-01` en el último origen y `2024-04` en los primeros orígenes):** en el pronóstico a 12 meses desde el último origen (`2025-02` $\to$ `2025-03` a `2026-02`, `figuras/pronosticos.png`), el peor error de los tres modelos ocurre en **enero de 2026 (`2026-01`, lluvia real = `315.2 mm`)**:
   - En enero de 2024 hubo una sequía extrema (`15.0 mm`) y enero de 2025 siguió siendo inusualmente seco (`53.8 mm`), mientras que febrero de 2025 tuvo un salto húmedo (`464.0 mm`, $z_{\text{2025-02}} = +313.6\text{ mm}$).
   - Al reconstruir el nivel como $\hat{y}_{\text{2026-01}} = y_{\text{2025-01}} + \hat{z}_{\text{2026-01}} = 53.8 + \hat{z}_{\text{2026-01}}$, tanto **`SVR`** como **`LSTM (16 ocultas)`** arrastran diferencias recursivas negativas tras el pico de febrero y pronostican lluvias físicamente imposibles: **`-48.9 mm` en `SVR` (error `-364.1 mm`, RMSE del último origen `162.61 mm`)** y **`-12.3 mm` en `LSTM` (error `-327.5 mm`, RMSE del último origen `152.99 mm`)**.
   - En cambio, el **`SARIMAX`** ($\Theta_1 = -0.853$) descuenta el $85.3\,\%$ de la sequía anormal de `2025-01` y pronostica **`135.6 mm` en `2026-01` (error `-179.6 mm`, RMSE del último origen `102.48 mm`)**, evitando el colapso.
4. **Inestabilidad entre semillas:** en la `LSTM (16 ocultas)` el simple cambio de semilla hace variar el RMSE en un rango de **`26.08 mm`** (de `133.66 mm` con semilla `11` a `159.74 mm` con semilla `1`, desviación de `11.31 mm`), mientras que el modelo clásico es determinístico dado el máximo verosímil.

## 6. Conclusión

El grupo **pondría en uso el modelo clásico del Miniproyecto 1 —`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` (o su versión `SARIMA(0, 0, 0)(0, 1, 1, 12)` pura de `2` parámetros, dado que la exógena de días del mes solo modifica febreros bisiestos)**— y, como alternativa de aprendizaje profundo únicamente si se exigiera una arquitectura neuronal, la **`LSTM (16 ocultas), 37 épocas`** (o la **`Conv1D (8 canales), 21 épocas`** por su bajoMAPE de `43.11 %` con solo `41` parámetros).

Frente al modelo clásico (`RMSE = 113.79 mm`, `MAE = 91.75 mm`, `3` parámetros, `3.5 s`), elegir la mejor red neuronal (**`LSTM (16 ocultas), 37 épocas`**, `RMSE = 137.84 mm`, `MAE = 99.89 mm`) implica pagar un costo desfavorable en las tres dimensiones operativas y un mayor error (`+24.05 mm` en RMSE y `+8.14 mm` en MAE):
- **En datos:** el SARIMA/SARIMAX opera desde el mes `13` (solo pierde `12` meses por la diferencia estacional $D = 1$, aprovechando `122` observaciones efectivas), mientras que la tabla supervisada y las ventanas de la LSTM con `OBJETIVO = "dif12"` y `12` rezagos **pierden `24` meses de historia** (todo `2015` y `2016`, dejando solo `86` a `98` filas por origen), un sacrificio crítico en series hidrológicas mensuales cortas.
- **En tiempo de entrenamiento y estabilidad:** aunque un *backtest* de la LSTM con `37` épocas fijas en CPU tarda **`3.0 s`** (similar a los **`3.5 s`** del SARIMAX y más rápido que los **`6.2 s`** de Random Forest), elegir sus épocas con validación previa y estabilizar su varianza entre semillas ($\pm 11.31\text{ mm}$ de RMSE en `5` semillas) multiplica por `5` el tiempo de cómputo ($\approx 15\text{ s}$).
- **En explicabilidad:** el modelo clásico se resume en un único parámetro interpretable ($\Theta_1 = -0.853$: cada mes se pronostica como el promedio histórico de ese mes más el $14.7\,\%$ de la anomalía del año anterior) y garantiza niveles coherentes con la climatología de la cuenca de Pance, mientras que la LSTM distribuye su decisión en **`1 233` parámetros opacos** de compuertas recurrentes que, ante secuencias atípicas en modo recursivo, pueden predecir precipitaciones negativas (`-12.3 mm` en `2026-01`).
