# Behavioral Portfolio Advisor

Proyecto final (Proyecto A): un dashboard en Streamlit que perfila a un inversionista con Prospect Theory, le construye un portafolio con un MILP (Gurobi) y simula durante 10 años cómo su propio comportamiento afecta su patrimonio. El problema está planteado con el marco de Powell.

Dashboard en línea: https://rodrigoveytiahdez-hue-behavioral-portfolio-advisor-app-1gmcql.streamlit.app/

## Cómo correrlo

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/rodrigoveytiahdez-hue/behavioral-portfolio-advisor.git
cd behavioral-portfolio-advisor
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

- **Gurobi:** `pip install gurobipy` trae una licencia limitada que alcanza para este modelo (unas 130 variables). No hace falta una licencia académica.
- **Datos:** la app descarga ~7 años de precios desde Yahoo Finance. Si no hay internet o Yahoo no responde, usa los CSV de `data/`.
- **Tiempo:** la simulación tarda 1-3 minutos, sobre todo porque la política CFA vuelve a resolver el MILP cada año en cada uno de los 500 escenarios. Si se repite con los mismos datos, el resultado sale de caché.

### Otros comandos

```bash
python datos.py        # vuelve a descargar precios y actualiza data/
python motor.py        # resuelve el MILP para un cliente de prueba (lambda = 2.5)
python simulacion.py   # corre la evaluación completa e imprime la tabla de resultados
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `activos.json` | Catálogo de 24 ETFs en 6 clases de activo (acciones EUA, internacionales, sectoriales, renta fija, bienes raíces, commodities) |
| `datos.py` | Descarga precios diarios con yfinance y los convierte en rendimientos mensuales (quita el mes en curso si está incompleto) |
| `cuestionario.py` | 10 preguntas → λ (aversión a pérdidas), α y β (curvatura) y banderas de sesgo (panic selling, performance chasing) |
| `motor.py` | MILP mean-CVaR (Rockafellar–Uryasev) en Gurobi |
| `simulacion.py` | Simulador de Powell, las 6 políticas, tuning y evaluación |
| `app.py` | Dashboard en Streamlit |

## Parte 1: Perfil del cliente

- **λ:** loterías 50/50 de ganar G o perder $5,000, con G = $10k, $15k y $20k. λ se aproxima con la razón G/L a la que el cliente empieza a aceptar.
- **α y β:** equivalentes ciertos. Con v(x) = x^α y una lotería 50/50 entre un premio P y 0, el equivalente cierto CE cumple α = ln(0.5) / ln(CE / P). α sale de dos preguntas de ganancias (P = $10k y $20k) y β de una de pérdidas.
- **Sesgos:** 4 preguntas de escenario. Su puntaje define el umbral de pánico (15%, 20% o 30% de drawdown) y la inclinación a perseguir ganadores (10%, 20% o 30%).

## Parte 2: Portafolio (MILP)

Variables: pesos `w_i` y binarias `z_i`. Restricciones: los pesos suman 1; sin ventas en corto; a lo más K activos; `0.05·z_i ≤ w_i ≤ 0.30·z_i`; máximo 40% por sector, 50% por clase de activo y 20% para activos alternativos (commodities y bienes raíces).

Objetivo: max μᵀw − θ(λ)·CVaR₉₅(w). La penalización θ va de 0.05 (λ = 1.5) a 0.50 (λ = 4.5), así que un cliente más averso a pérdidas recibe un portafolio con menos CVaR.

## Parte 3: Modelo de Powell

| Elemento | Definición |
|---|---|
| Estado S_t | Riqueza W_t ($), pesos actuales w_t, riqueza máxima a la fecha (para el drawdown), rendimientos recientes, si el cliente está en efectivo |
| Decisión x_t | Pesos objetivo del mes, es decir, los trades. Se decide al inicio del mes con el estado conocido |
| Información exógena W_{t+1} | Rendimientos mensuales de los 24 ETFs (block bootstrap con bloques de 3 a 6 meses) |
| Transición | W_{t+1} = (W_t − 0.001·W_t·Σ\|x_t − w_t\|)·(1 + x_tᵀr_{t+1}); los pesos se mueven con los rendimientos |
| Objetivo | max E[v((W_T / W_0) − 1)], donde v es la función de valor de Prospect Theory con λ, α y β del cliente. T = 120 meses |

### Políticas

| Política | Clase | Parámetros ajustables |
|---|---|---|
| Buy & Hold | Baseline | — |
| Band Rebalancing | PFA | Ancho de banda (ajustado en simulación) |
| Lookahead | DLA | Exposición {50%, 75%, 100%}, horizonte de 12 meses, 200 escenarios internos |
| Re-optimización anual | CFA | Ventana de 60 meses, frecuencia de 12 meses, θ(λ) |
| Panic Selling | Conductual | Umbral de drawdown d, recuperación r = 10% |
| Performance Chasing | Conductual | Inclinación hacia los 3 ganadores de los últimos 12 meses |

### Evaluación

- Todas las políticas usan los mismos 500 escenarios (common random numbers).
- Semillas separadas: tuning = 111, evaluación final = 999, escenarios internos del DLA = 777.
- Se reportan media, mediana, percentiles 5 y 95, desviación estándar e IC 95% de la riqueza final y del valor prospect, además de la diferencia pareada contra Buy & Hold con su IC 95%.

## Limitaciones

- El MILP y el bootstrap usan los mismos ~7 años de datos (2020-2026), un periodo muy bueno para tecnología y oro. Los resultados son in-sample y probablemente optimistas.
- El portafolio óptimo es sensible a la muestra: agregar o quitar un mes puede cambiar la selección de activos.
- El efectivo rinde 0% en las políticas que salen del mercado.
