# Informe del miniproyecto 2

## 1. La serie y el modelo a superar

La serie corresponde a la **precipitación mensual acumulada ($\text{mm/mes}$)** en la estación **CDEA El Topacio** (código `2629020601`), ubicada en la cuenca del río Pance (Cali, Valle del Cauca), suministrada por la **Corporación Autónoma Regional del Valle del Cauca (CVC)**. Abarca de **enero de 2015 a febrero de 2026 (`134` observaciones mensuales continuas)**, con un régimen bimodal andino (máximo estacional en abril con $+118.6\text{ mm}$ y mínimo en agosto con $-125.1\text{ mm}$) y un rango observado entre **$5.7\text{ mm}$** (`2019-08`) y **$577.0\text{ mm}$** (`2024-04`). Del Miniproyecto 1 se mantiene la misma función `preparar`: **no hay meses faltantes (`0` huecos)**, duplicados ni valores negativos, y **se conservan los `7` meses atípicos** detectados por el rango intercuartílico del residuo de la descomposición STL robusta ($[-174.5,\ 172.7]\text{ mm}$: `2025-12`, `2022-06`, `2024-04`, `2023-03`, `2022-05`, `2023-10` y `2022-02`), dejando señalados los meses con totales diarios a las 07:00 como posible artefacto de medición sin alterar la serie para no introducir fuga de información.

![La serie](../figuras/serie.png)

Del Miniproyecto 1 se traslada su mejor modelo, el **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`**. **En palabras: la lluvia de un mes se pronostica a partir de la del mismo mes del año anterior, descontando la mayor parte de lo que ese mes tuvo de inusual, con un ajuste por el número de días del mes.** En ecuación:

$$y_t = y_{t-12} + \beta\,(x_t - x_{t-12}) + \varepsilon_t + \Theta_1\,\varepsilon_{t-12},$$

donde $y_t$ es la lluvia del mes, $x_t$ sus días del calendario (se conocen de antemano, así que no hay fuga) y $\varepsilon_t$ el error del pronóstico. Con toda la serie, $\Theta_1 = -0.853$: el modelo descuenta el 85 % del error de hace un año y se queda cerca de lo normal para ese mes. El ajuste por días ($\beta = -100.9$ mm por día de diferencia) solo actúa en febreros bisiestos y en el febrero siguiente, porque los demás meses tienen los mismos días que el año anterior. Se estiman **`3` parámetros** ($\beta$, $\Theta_1$ y la varianza del error); el ajuste se hace sobre la lluvia dividida entre 100, solo por estabilidad numérica. Al pasar por el mismo *backtest* de `comun.py` (`13` orígenes desde `2024-02` hasta `2025-02`, de 1 a 12 meses adelante, sobre los últimos `24` meses, `2024-03` a `2026-02`), sus métricas **coinciden exactamente con las del Miniproyecto 1**:

| Modelo clásico de referencia | RMSE (mm) | MAE (mm) | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`** | **113.79** | **91.75** | 54.31 | 3 | 2.3 |
| `SARIMA(0, 0, 0)(0, 1, 1, 12)` (sin exógena, ref. MP1) | 117.05 | 93.35 | 54.67 | 2 | — |
| `estacional ingenuo` (piso de referencia) | 154.93 | 112.39 | **43.90** | 0 | 0.0 |

No hubo ningún cambio en las métricas al trasladarlo (`RMSE = 113.79 mm`, `MAE = 91.75 mm`, `MAPE = 54.31 %`), y el guion imprime también la referencia del `SARIMA(0, 0, 0)(0, 1, 1, 12)` puro ($y_t - y_{t-12} = \varepsilon_t + \Theta_1 \varepsilon_{t-12}$, `RMSE = 117.05 mm`, `MAE = 93.35 mm`, `2` parámetros), mostrando que casi toda la reducción del error frente al `estacional ingenuo` (`-26.6 %` en RMSE y `-18.4 %` en MAE) proviene del término estacional $\Theta_1 \approx -0.853$, que amortigua los choques pluviales del año anterior hacia el promedio histórico de cada mes.

## 2. La tabla supervisada

**Por qué una tabla.** Los modelos de aprendizaje automático y las redes aprenden de forma **supervisada**: necesitan ejemplos que traigan las entradas ($X$) y la respuesta correcta ($y$). Una serie de tiempo es una sola columna, sin entradas. Por eso fabricamos las entradas con el pasado de la misma serie: **cada fila es un mes, sus columnas son lo que se sabía antes de ese mes y la respuesta es lo que pasó ese mes.** Así, cada mes de la historia se vuelve un ejemplo resuelto del que el modelo aprende. Usamos el mismo diseño de tabla de la sesión 7 del curso.

**Qué aprenden: la diferencia con el año anterior (`OBJETIVO = "dif12"`).** En lugar de la lluvia, cada modelo aprende cuánto cambió respecto al mismo mes del año anterior. Después se le suma ese mes para volver a milímetros:

$$z_t = y_t - y_{t-12}, \qquad \hat y_t = y_{t-12} + \hat z_t.$$

Lo escogimos frente a la otra opción de la plantilla, aprender la lluvia directa (`OBJETIVO = "nivel"`), por dos razones:
- **Quita el ciclo anual.** La prueba ADF del Miniproyecto 1 dice que la serie original no es estacionaria ($p = 0.2092$) y que la diferencia sí lo es ($p = 0.0033$).
- **Funciona mejor con las redes y en los meses secos.** Con `nivel` los árboles mejoran, porque la lluvia no tiene tendencia y el techo de lo ya visto no compromete. Pero las redes empeoran y todos los modelos se van a un MAPE de `66` a `93 %`. La comparación completa está en la sección 5.

**Las columnas.** La función `tabla_supervisada` arma **`14` columnas**, todas con datos anteriores al mes que se pronostica:

| Columnas | Qué son |
|---|---|
| `rezago_1` a `rezago_12` | La diferencia $z$ de cada uno de los 12 meses anteriores |
| `mes` | El mes del año (1 a 12) |
| `media_3` | El promedio de $z$ en los 3 meses anteriores |

Se pierden los dos primeros años: 2015 por la diferencia y 2016 por el rezago 12. Quedan **`110` filas**, de `2017-01` a `2026-02`.

**Prueba de que no hay fuga.** Si una columna usara el dato del mes que se quiere pronosticar, sería como darle al modelo la respuesta dentro del enunciado. La función `comprobar_sin_fuga` lo descarta: le suma 10 000 mm a cada uno de los `134` meses, rehace la tabla y verifica que la fila de ese mismo mes no cambie. Si cambiara, el guion se detendría. Pasa en los 134 meses:

```text
Tabla supervisada: 134 filas totales, 110 completas (2017-01 a 2026-02), 14 columnas: [...]
Sin fuga verificado: ninguna de las 134 filas cambia al alterar y(t) o z(t) en el mes que predice
```

**Otros tres cuidados contra la fuga:**
1. **El escalado se ajusta en cada origen solo con su entrenamiento** (de `86` filas en el primer origen a `98` en el último), dentro de cada modelo.
2. **Los hiperparámetros se eligen sin mirar el *backtest*:** con `TimeSeriesSplit` sobre los meses anteriores al primer origen, sin barajar el tiempo.
3. **La `Conv1D` no mira hacia adelante** (`padding=0`).

**El pronóstico a 12 meses es recursivo.** Se pronostica el mes 1. Ese pronóstico se escribe en la serie como si fuera un dato y sirve de rezago para el mes 2, y así hasta el mes 12. Por eso el error crece con el horizonte.

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
- **Conexión con la ACF y PACF del Miniproyecto 1:** en el Miniproyecto 1, los únicos rezagos dentro del primer año que salían de la banda de confianza ($\pm 0.177$) en la ACF y la PACF de $z_t = y_t - y_{t-12}$ eran el **rezago `12`** ($\text{ACF}_{12} = -0.35$, $\text{PACF}_{12} = -0.41$) y el **rezago `7`** ($\text{ACF}_7 = +0.27$, $\text{PACF}_7 = +0.27$). Ambos ensambles descubren por sí solos que `rezago_12` es el predictor fundamental (por la reversión a la media interanual inducida al compartir $y_{t-12}$ con signo opuesto en $z_t = y_t - y_{t-12}$ y $z_{t-12} = y_{t-12} - y_{t-24}$). El rezago 7, en cambio, pesa algo más que el promedio de las 14 columnas (`7.14 %`) en XGBoost (`8.14 %`) y algo menos en Random Forest (`4.76 %`), así que su coincidencia con la ACF es débil. El hecho de que `mes` pese casi cero (`1.51 %` y `0.00 %`) confirma que la diferencia estacional `dif12` ya eliminó el ciclo anual determinístico.

En el *backtest* común sobre los últimos 24 meses, los cuatro modelos de aprendizaje automático obtienen:

| Modelo de aprendizaje automático | RMSE (mm) | MAE (mm) | MAPE (%) | Parámetros / nodos | Segundos |
|---|---|---|---|---|---|
| **`SVR` (`C=100, epsilon=0.5`)** | **150.93** | **113.94** | 55.25 | 1 471 | 1.4 |
| `XGBoost` (`max_depth=2, lr=0.01`) | 154.27 | 117.69 | **51.23** | 1 374 | 1.5 |
| `Random Forest` (`max_depth=10, leaf=5`) | 158.36 | 121.56 | 58.72 | 1 904 | 2.6 |
| `KNN` (`n_neighbors=5`) | 165.65 | 132.21 | 69.75 | 1 372 | 2.7 |

**Ganador entre los cuatro de aprendizaje automático:** **`SVR`** con **`RMSE = 150.93 mm`** y **`MAE = 113.94 mm`**, seguido de **`XGBoost`** (`RMSE = 154.27 mm`). Tanto `SVR` como `XGBoost` superan en RMSE al `estacional ingenuo` (`154.93 mm`), aunque por poco, mientras que `Random Forest` (`158.36 mm`) y `KNN` (`165.65 mm`) quedan por encima del piso ingenuo. En MAE, el ingenuo (`112.39 mm`) queda por delante de los cuatro: `SVR` y `XGBoost` cometen menos errores muy grandes que el ingenuo, pero en promedio fallan algo más.

## 4. Aprendizaje profundo

Las tres redes aprenden de la **misma tabla de la sección 2**, con el objetivo `dif12` y el pronóstico recursivo. Todas usan la clase `Red` con la misma receta: optimizador Adam (`lr = 0.01`), pérdida MSE, datos estandarizados con el entrenamiento de cada origen y hasta `300` épocas.

**Las tres redes.**

| Red | Cómo lee los datos | Parámetros |
|---|---|---|
| **MLP** | Recibe las 14 columnas juntas y las combina en una capa oculta. Se probaron dos tamaños: 8 y 32 neuronas | 129 y 513 |
| **LSTM (16 ocultas)** | Lee solo los 12 rezagos, **en orden**, un mes por paso, del más antiguo al más reciente. Sus compuertas deciden qué recordar | 1 233 |
| **Conv1D (8 canales)** | Recorre los 12 rezagos con una ventana de 3 meses, busca 8 patrones locales y promedia dónde aparecen | **41** |

**Por qué la tercera red es la Conv1D.** La elegimos por parsimonia: con `74` filas de entrenamiento, es la red con menos parámetros. La función `comparar_tercera_red` deja a la vista las otras candidatas:
- **Conv1D con *flatten* (`113` parámetros).** Le da un peso propio a cada posición del año. Tuvo menor error de validación (`1.4377` frente a `1.7224`), pero en el *backtest* empeora de `152.49` a `175.03 mm`. Con solo 12 meses de validación preferimos la red más pequeña. El *backtest* no se usó para escoger.
- **Híbrido SARIMAX + MLP sobre sus residuos (`244` parámetros).** No tiene qué aprender: el residuo del SARIMAX ya es ruido blanco (Ljung-Box $p(12) = 0.4689$). Obtiene `116.23 mm`, peor que el SARIMAX solo.
- **Transformer de una capa (`2 273` parámetros).** La atención reparte casi el mismo peso a los 12 meses (entre `0.072` y `0.117`, cerca de $1/12 \approx 0.083$): con tan pocas ventanas, casi promedia. En el *backtest* empata con la Conv1D (`152.00` frente a `152.49 mm`), con 55 veces más parámetros.

**Cuántas épocas entrenar.** Se eligieron sin tocar el *backtest*: cada red entrena con las primeras **`74` filas** (`2017-01` a `2023-02`) y se valida con los **`12` meses siguientes** (`2023-03` a `2024-02`). La mejor época es la de menor error de validación.

![Curvas de pérdida](../figuras/perdida.png)

| Red | Parámetros | Mejor época | Error de validación en esa época | Error de entrenamiento en la época 300 |
|---|---|---|---|---|
| MLP (8 ocultas) | 129 | 14 | 1.5716 | 0.0005 |
| **MLP (32 ocultas)** | 513 | **11** | **1.4099** | 0.0000 |
| **LSTM (16 ocultas)** | 1 233 | **37** | 1.4656 | 0.0002 |
| **Conv1D (8 canales)** | 41 | **21** | 1.7224 | 0.8084 |

Los errores están en escala estandarizada, no en milímetros.

**Lectura de las curvas:**
- **Las redes grandes memorizan muy pronto.** Los dos MLP tocan su mejor validación entre las épocas 11 y 14. Después, su error de entrenamiento llega casi a cero, porque memorizaron las 74 filas, y el de validación sube.
- **La LSTM aprende más despacio** y llega a su mejor punto en la época 37, pero también termina memorizando.
- **La Conv1D no puede memorizar:** con 41 parámetros, su error de entrenamiento se queda en 0.81.
- **Del MLP pasa el de 32 neuronas**, que tuvo menor validación (`1.4099` frente a `1.5716`).

**Resultado y semillas.** La mejor red en el *backtest* es la **LSTM con 37 épocas** (`RMSE = 137.37 mm`, `MAE = 99.40 mm`, `MAPE = 44.34 %`). La semilla fija los pesos con que arranca la red, así que repetimos su *backtest* con cinco semillas (`resultados/semillas.csv`):

| Semilla | RMSE (mm) | MAE (mm) | MAPE (%) |
|---|---|---|---|
| `2026` | 137.37 | 99.40 | 44.34 |
| `1` | 157.59 | 122.97 | 60.16 |
| `7` | 152.96 | 116.18 | 56.38 |
| `42` | 138.58 | 105.35 | 49.58 |
| `11` | 132.92 | 103.36 | 52.59 |
| **Media** | **143.88** | **109.45** | **52.61** |
| **Desviación estándar** | **10.74** | **9.78** | **6.10** |

El RMSE va de `132.92` a `157.59 mm`. El SARIMAX (`113.79 mm`) le gana a la media por `30.09 mm`, casi tres desviaciones (`2.80`), y a la mejor semilla por `19.13 mm`. La diferencia es **claramente mayor que la variación entre semillas**: la red no pierde por mala suerte en los pesos iniciales.

## 5. La tabla y la elección

La tabla completa generada por `python miniproyecto.py` en `resultados/metricas.csv`, ordenada de menor a mayor RMSE y sin modificar `comun.py`, es:

| Modelo | RMSE | MAE | MAPE (%) | Parámetros | Segundos |
|---|---|---|---|---|---|
| **`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas`** | **113.79** | **91.75** | 54.31 | **3** | 2.3 |
| `LSTM (16 ocultas), 37 épocas` | 137.37 | 99.40 | 44.34 | 1 233 | 1.4 |
| `SVR` | 150.93 | 113.94 | 55.25 | 1 471 | 1.4 |
| `Conv1D (8 canales), 21 épocas` | 152.49 | 109.64 | **43.11** | 41 | **0.8** |
| `MLP (32 ocultas), 11 épocas` | 154.03 | 113.47 | 66.20 | 513 | **0.8** |
| `XGBoost` | 154.27 | 117.69 | 51.23 | 1 374 | 1.5 |
| `estacional ingenuo` | 154.93 | 112.39 | 43.90 | 0 | 0.0 |
| `Random Forest` | 158.36 | 121.56 | 58.72 | 1 904 | 2.6 |
| `KNN` | 165.65 | 132.21 | 69.75 | 1 372 | 2.7 |

Los segundos cambian un poco entre corridas y entre equipos; los de esta tabla son de una corrida en CPU, en la que el guion completo tardó `57 s`.

**¿Y si los modelos aprendieran la lluvia directa (`OBJETIVO = "nivel"`)?** Es la otra opción de la plantilla (sección 2). La probamos con los mismos modelos y el mismo *backtest*:

| Modelo | RMSE con `nivel` | RMSE con `dif12` (el usado) |
|---|---|---|
| `XGBoost` | 133.45 | 154.27 |
| `Random Forest` | 135.03 | 158.36 |
| `KNN` | 141.18 | 165.65 |
| `LSTM (16 ocultas)` | 154.80 | **137.37** |
| `Conv1D (8 canales)` | 154.32 | 152.49 |
| `SARIMAX` (no usa la tabla) | **113.79** | **113.79** |

Con `nivel`, los árboles y KNN mejoran entre 20 y 24 mm: la lluvia no tiene tendencia, así que promediar lluvias ya vistas no los limita, y `XGBoost` (`133.45 mm`) quedaría por delante de la `LSTM`. Pero las redes empeoran (la `LSTM` llega a `MAE = 134.88 mm`) y todos los modelos quedan con un MAPE de entre `66 %` y `93 %`, muy por encima del `43.90 %` del estacional ingenuo: en los meses secos pronostican la lluvia de un mes normal. Como la plantilla usa un solo objetivo para todos los modelos, nos quedamos con `dif12`. **Con ninguno de los dos objetivos el SARIMAX deja de ganar.**

**El peor mes de cada modelo.** El guion guarda en `resultados/peor_mes.csv` el mes del *backtest* con mayor error absoluto medio, tomando todos los orígenes que pronostican ese mes:

| Modelo | Peor mes | Lluvia real (mm) | Error absoluto medio (mm) | Pronósticos de ese mes |
|---|---|---|---|---|
| `SARIMAX` | `2024-04` | 577.0 | 303.1 | 2 |
| `LSTM (16 ocultas), 37 épocas` | `2024-04` | 577.0 | 361.2 | 2 |
| `SVR` | `2024-04` | 577.0 | 345.2 | 2 |
| `Conv1D (8 canales), 21 épocas` | `2024-03` | 126.0 | 346.0 | 1 |
| `MLP (32 ocultas), 11 épocas` | `2025-02` | 464.0 | 345.6 | 12 |
| `XGBoost` | `2024-04` | 577.0 | 431.4 | 2 |
| `estacional ingenuo` | `2024-03` | 126.0 | 348.6 | 1 |
| `Random Forest` | `2024-04` | 577.0 | 447.2 | 2 |
| `KNN` | `2024-04` | 577.0 | 373.0 | 2 |

Seis de los nueve modelos fallan más en **abril de 2024**, el récord de la serie (`577.0 mm`): ninguno lo ve venir, y el que menos se equivoca es el SARIMAX. Los dos árboles se equivocan más (`431` a `447 mm`) porque no pronostican por encima de las diferencias que ya vieron. La `Conv1D` y el estacional ingenuo fallan más en marzo de 2024 (`126.0 mm`): el ingenuo copia el marzo de 2023, uno de los meses atípicos (`474.6 mm`), y la `Conv1D` reconstruye el nivel sobre ese mismo marzo con una diferencia pronosticada casi nula. El `MLP` falla más en febrero de 2025 (`464.0 mm`), un salto húmedo que pronostican los 12 orígenes.

![Pronósticos desde el último origen](../figuras/pronosticos.png)

**¿Por qué ninguna red neuronal ni modelo de aprendizaje automático supera al clásico en nuestra serie?** Lo explican cuatro datos concretos de la serie de Pance:

1. **Desproporción entre número de parámetros y meses disponibles:** en los `13` orígenes del *backtest*, la tabla supervisada solo dispone de **`86` a `98` filas completas** de entrenamiento (`74` filas al descontar validación). Mientras el `SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` estima **`3` parámetros** (`2` en el SARIMA puro, a razón de $\approx 49$ observaciones por parámetro), el `MLP (32 ocultas)` ajusta **`513` pesos** ($5.2$ parámetros por fila) y la `LSTM (16 ocultas)` ajusta **`1 233` pesos** ($12.6$ parámetros por fila). Como muestran las curvas de pérdida (`figuras/perdida.png`), a partir de la época `11` en el MLP y de la `37` en la LSTM las redes memorizan el ruido meteorológico de entrenamiento (`MSE` $\to 0.0000$).
2. **Estructura parsimoniosa de la dependencia temporal:** como demostraron la ACF/PACF del Miniproyecto 1 y las importancias de variables (`figuras/importancia.png`, donde `rezago_12` concentra `43.15 %` en Random Forest y `21.02 %` en XGBoost), la única señal lineal fuerte en $z_t = y_t - y_{t-12}$ es la autocorrelación negativa en el rezago 12 ($\rho_{12} = -0.35$) producida por el rebote hacia la media tras un año anómalo. El parámetro $\Theta_1 = -0.853$ del SARIMA captura exactamente esa relación con una sola ecuación en diferencias, mientras que las redes y árboles intentan combinar `12` a `14` entradas ruidosas y propagan esos errores a lo largo de los `12` pasos recursivos.
3. **El mes crítico en que fallan los modelos y el problema de los récords climáticos (`2026-01` en el último origen y `2024-04 = 577.0 mm` en los primeros orígenes):** los modelos entrenados con pocas filas no pronostican bien los récords o los rebotes extremos que no aparecen en su entrenamiento (como muestra la tabla del peor mes, seis de los nueve modelos fallan más en el récord de `2024-04`). En el pronóstico a 12 meses desde el último origen (`2025-02` $\to$ `2025-03` a `2026-02`, `figuras/pronosticos.png`), el peor error de los tres modelos ocurre en **enero de 2026 (`2026-01`, lluvia real = `315.2 mm`)**:
   - En enero de 2024 hubo una sequía extrema (`15.0 mm`) y enero de 2025 siguió siendo inusualmente seco (`53.8 mm`), mientras que febrero de 2025 tuvo un salto húmedo (`464.0 mm`, $z_{\text{2025-02}} = +313.6\text{ mm}$).
   - Al reconstruir el nivel como $\hat{y}_{\text{2026-01}} = y_{\text{2025-01}} + \hat{z}_{\text{2026-01}} = 53.8 + \hat{z}_{\text{2026-01}}$, tanto **`SVR`** como **`LSTM (16 ocultas)`** arrastran diferencias recursivas negativas tras el pico de febrero y llegan a lluvias físicamente imposibles (`-48.9 mm` en `SVR` y `-12.3 mm` en `LSTM`). Por eso `a_nivel` recorta todo pronóstico en `0 mm`: con el recorte ambos pronostican **`0.0 mm` (error `-315.2 mm`)**, con RMSE del último origen de **`153.87 mm` en `SVR`** y **`150.83 mm` en `LSTM`** (sin recorte eran `162.61 mm` y `152.99 mm`). El recorte nunca empeora un pronóstico y mejora un poco a todos los modelos con `dif12` (por ejemplo, la `LSTM` baja de `137.84` a `137.37 mm` y `KNN` de `168.96` a `165.65 mm`), pero no cambia la conclusión: el modelo sigue sin ver venir el enero lluvioso.
   - En cambio, el **`SARIMAX`** ($\Theta_1 = -0.853$) descuenta el $85.3\,\%$ de la sequía anormal de `2025-01` y pronostica **`135.6 mm` en `2026-01` (error `-179.6 mm`, RMSE del último origen `102.48 mm`)**, evitando el colapso.
4. **Inestabilidad entre semillas:** en la `LSTM (16 ocultas)` el simple cambio de semilla hace variar el RMSE en un rango de **`24.67 mm`** (de `132.92 mm` con semilla `11` a `157.59 mm` con semilla `1`, desviación de `10.74 mm`), mientras que el modelo clásico es determinístico dado el máximo verosímil.

## 6. Conclusión

El grupo **pondría en uso el modelo clásico del Miniproyecto 1 —`SARIMAX(0, 0, 0)(0, 1, 1, 12) + exógenas` (o su versión `SARIMA(0, 0, 0)(0, 1, 1, 12)` pura de `2` parámetros, dado que la exógena de días del mes solo modifica febreros bisiestos)**— y, como alternativa de aprendizaje profundo únicamente si se exigiera una arquitectura neuronal, la **`LSTM (16 ocultas), 37 épocas`** (o la **`Conv1D (8 canales), 21 épocas`** por su bajo MAPE de `43.11 %` con solo `41` parámetros).

Frente al modelo clásico (`RMSE = 113.79 mm`, `MAE = 91.75 mm`, `3` parámetros, `2.3 s`), elegir la mejor red neuronal (**`LSTM (16 ocultas), 37 épocas`**, `RMSE = 137.37 mm`, `MAE = 99.40 mm`) implica pagar un costo desfavorable en las tres dimensiones operativas y un mayor error (`+23.58 mm` en RMSE y `+7.65 mm` en MAE):
- **En datos:** el SARIMA/SARIMAX opera desde el mes `13` (solo pierde `12` meses por la diferencia estacional $D = 1$, aprovechando `122` observaciones efectivas), mientras que la tabla supervisada y las ventanas de la LSTM con `OBJETIVO = "dif12"` y `12` rezagos **pierden `24` meses de historia** (todo `2015` y `2016`, dejando solo `86` a `98` filas por origen), un sacrificio crítico en series hidrológicas mensuales cortas.
- **En tiempo de entrenamiento y estabilidad:** aunque un *backtest* de la LSTM con `37` épocas fijas en CPU tarda **`1.4 s`** (del orden de los **`2.3 s`** del SARIMAX y los **`2.6 s`** de Random Forest), elegir sus épocas con validación previa y estabilizar su varianza entre semillas ($\pm 10.74\text{ mm}$ de RMSE en `5` semillas) multiplica por `5` el tiempo de cómputo ($\approx 7\text{ s}$).
- **En explicabilidad:** el modelo clásico se resume en un único parámetro interpretable ($\Theta_1 = -0.853$: cada mes se pronostica como el promedio histórico de ese mes más el $14.7\,\%$ de la anomalía del año anterior) y garantiza niveles coherentes con la climatología de la cuenca de Pance, mientras que la LSTM distribuye su decisión en **`1 233` parámetros opacos** de compuertas recurrentes que, ante secuencias atípicas en modo recursivo, pueden predecir precipitaciones negativas (`-12.3 mm` en `2026-01` antes del recorte en `0 mm`).
- **Modelo local frente a modelo global — ¿cambiaría la elección si tuviéramos que pronosticar mil series como la nuestra?:** **sí cambiaría.** Para **una sola serie mensual corta** (la estación CDEA El Topacio aislada, con `86`–`98` filas supervisadas), un **modelo local** parsimonioso como `SARIMAX` (`3` parámetros) es imbatible porque los modelos con cientos o miles de parámetros sobreajustan. Sin embargo, si la CVC tuviera que pronosticar simultáneamente **`1 000` estaciones pluviométricas** de la red hidroclimatológica andina, apilar las `1 000` series estandarizadas en una única tabla supervisada con covariables de cuenca y altitud (**modelo global**, Montero-Manso y Hyndman, 2021) proporcionaría **`~110 000` ventanas de entrenamiento**. En ese régimen ($< 0.02$ parámetros por ventana), un **`XGBoost`** o una red global (**`LSTM` / `Transformer`**) aprendería patrones regionales compartidos (como anomalías de El Niño y La Niña entre cuencas vecinas) sin sobreajustar y evitaría ajustar, diagnosticar y mantener `1 000` modelos SARIMA individuales.
