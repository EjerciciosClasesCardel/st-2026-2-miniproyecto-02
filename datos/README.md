# datos

La serie del grupo, la misma del miniproyecto 1, con la ficha que se llenó entonces.

| Campo | Valor |
|---|---|
| Archivo | `pance_mensual.csv` |
| Qué mide | Precipitación mensual acumulada (mm) en la estación CDEA El Topacio (código 2629020601), cuenca del río Pance (Cali, Valle del Cauca) |
| Fuente | Corporación Autónoma Regional del Valle del Cauca (CVC), red de telemetría hidrometeorológica ([cvc.gov.co](https://www.cvc.gov.co)). Datos suministrados por la CVC por correo electrónico |
| Licencia | Uso académico autorizado por la CVC; se cita a la CVC como fuente. No es una licencia abierta |
| Frecuencia | mensual |
| Construcción | En el miniproyecto 1 se unieron los archivos `EXPORT_CPANCE_PRECIP_*.xls` (telemetría cada 10 minutos y, en días con falla del sensor, totales diarios a las 07:00), se quitaron duplicados y se sumó por mes. Un mes con menos del 80 % de días con datos habría quedado vacío; la regla no se activó |
| Periodo | 2015-01 a 2026-02 (134 observaciones) |
| Columna de fecha | `mes` (formato `AAAA-MM`) |
| Columna que se pronostica | `valor` (mm/mes) |

La serie de ejemplo (`air_passengers.csv`) se borró al pasar a la serie del grupo.
