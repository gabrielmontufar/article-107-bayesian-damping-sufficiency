# Nota de reproducción del bloque sintético

## Modelo y unidad de información

Se simulan ventanas independientes de decaimiento libre SDOF con `f_n = 1 Hz`, `ζ = 0.03`, amplitud unitaria, fase `0.35 rad`, cinco ciclos y `f_s = 50 Hz`. Cada ventana tiene ruido gaussiano independiente (`σ = 0.50`) y comparte el valor de amortiguamiento, mientras que amplitud y fase se ajustan por mínimos cuadrados condicionales para cada valor candidato de `ζ`. Por ello `N` significa número de ventanas independientes; el número de muestras se informa por separado y no se presenta como tamaño muestral efectivo.

La inferencia principal usa la verosimilitud perfilada (mínimos cuadrados condicionales para amplitud y fase) y cuadratura determinista en una rejilla de `ζ`, con tres priors truncados implícitos por el intervalo de trabajo: débil (`μ = 0.03, σ = 0.03`), informativo (`μ = 0.03, σ = 0.01`) y sesgado (`μ = 0.045, σ = 0.01`). La regla principal es `Λζ = z Sζ CV(ζ) / εR`, con `Sζ = 1` para la respuesta resonante `R(ζ) = 1/(2ζ)` y `εR = 0.10`. Las mismas observaciones de cada réplica se reutilizan entre priors; la semilla no cambia con el prior.

## Comparadores y validación

El mismo bloque calcula comparadores transparentes: perfil de verosimilitud de 95 % sin prior y estimador puntual MLE. La evaluación usa 100 réplicas por combinación (`N = 20, 50, 100, 300`; tres priors), con semillas derivadas de `20260728`. Se reportan media, desviación estándar, RMSE, cobertura empírica del IC95, tasas de aceptación y tasas de falsa aceptación/rechazo, junto con intervalos de Wilson para las tasas.

## Resultados verificables

Los resultados completos están en `05_results/synthetic_full_100_repaired/`. La cobertura empírica del IC95 queda entre 92–98 %. Con `N=20`, la regla principal acepta 0 % para el prior débil y presenta 92–94 % de falsos rechazos; a partir de 50 ventanas la aceptación es 100 %, con falsa aceptación de 1 % en `N=50` y 0 % en `N=100,300`. Para el prior débil, la media de `ζ` es 2.994 %, 2.994 %, 3.008 % y 2.999 % para `N=20,50,100,300`, con SD posterior media de 0.178 % a 0.046 %. La sensibilidad de `εR` en `N=100` produce aceptación de 8 %, 100 % y 100 % para `εR=5,10,20 %`, respectivamente.

Estos resultados no constituyen validación externa: son una calibración de comportamiento bajo el modelo generador declarado. El caso con aceleraciones reales debe analizarse por separado y no se incorporará como validación de amortiguamiento hasta que la identificación sea reproducible y auditable.
