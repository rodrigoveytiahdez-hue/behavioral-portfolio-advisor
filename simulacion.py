from dataclasses import dataclass

import numpy as np
import pandas as pd

from motor import optimizar_portafolio


# =========================================================
# RESULTADO DE UNA TRAYECTORIA
# =========================================================

@dataclass
class ResultadoTrayectoria:
    riqueza_final: float
    max_drawdown: float
    costos_transaccion: float
    numero_trades: int
    valor_prospect: float
    trayectoria: np.ndarray


# =========================================================
# PROSPECT THEORY
# =========================================================

def prospect_value(x, alpha, beta, lam):
    """
    Función de valor de Prospect Theory.

    x > 0  -> ganancias
    x < 0  -> pérdidas

    IMPORTANTE: x se mide como rendimiento relativo al punto
    de referencia (por ejemplo 0.25 = +25%), no en pesos.
    Si se usa en pesos y alpha != beta, el resultado cambia
    con el tamaño del capital, lo cual no tiene sentido.
    """

    x = np.asarray(x, dtype=float)

    return np.where(
        x >= 0,
        np.power(np.maximum(x, 0), alpha),
        -lam * np.power(np.maximum(-x, 0), beta)
    )


# =========================================================
# BLOCK BOOTSTRAP
# =========================================================

def generar_escenarios(
    retornos,
    n_escenarios=500,
    horizonte=120,
    bloque_min=3,
    bloque_max=6,
    seed=2026
):
    """
    Genera escenarios de rendimientos mensuales mediante
    block bootstrap.

    Los bloques contienen entre 3 y 6 meses consecutivos.
    Esto conserva parte de la estructura temporal observada.
    """

    rng = np.random.default_rng(seed)

    datos = retornos.to_numpy()

    n_meses, n_activos = datos.shape

    escenarios = np.zeros(
        (
            n_escenarios,
            horizonte,
            n_activos
        )
    )

    for n in range(n_escenarios):

        meses = []

        while len(meses) < horizonte:

            longitud = int(
                rng.integers(
                    bloque_min,
                    bloque_max + 1
                )
            )

            max_inicio = (
                n_meses - longitud
            )

            inicio = int(
                rng.integers(
                    0,
                    max_inicio + 1
                )
            )

            bloque = datos[
                inicio:inicio + longitud
            ]

            meses.extend(
                bloque.tolist()
            )

        escenarios[n] = np.array(
            meses[:horizonte]
        )

    return escenarios


# =========================================================
# PESOS
# =========================================================

def alinear_pesos(
    pesos,
    columnas
):
    """
    Convierte la cartera seleccionada por Gurobi
    en un vector compatible con todos los activos.
    """

    return np.array(
        [
            float(
                pesos.get(
                    ticker,
                    0.0
                )
            )
            for ticker in columnas
        ]
    )


# =========================================================
# TRANSACCIONES
# =========================================================

def aplicar_trade(
    riqueza,
    pesos_actuales,
    pesos_objetivo,
    costo=0.001
):
    """
    Transaction cost = 10 basis points
    sobre el valor negociado.

    10 bps = 0.10% = 0.001
    """

    turnover = np.abs(
        pesos_objetivo
        - pesos_actuales
    ).sum()

    costo_monetario = (
        riqueza
        * turnover
        * costo
    )

    riqueza_despues = (
        riqueza
        - costo_monetario
    )

    return (
        riqueza_despues,
        costo_monetario,
        turnover
    )


def actualizar_pesos(
    pesos,
    retornos_mes
):
    """
    Después de que cambian los precios,
    los pesos del portafolio también cambian.
    """

    crecimiento = (
        pesos
        * (1 + retornos_mes)
    )

    rendimiento_portafolio = (
        np.dot(
            pesos,
            retornos_mes
        )
    )

    denominador = (
        1 + rendimiento_portafolio
    )

    if denominador <= 0:
        return np.zeros_like(
            pesos
        )

    return (
        crecimiento
        / denominador
    )


# =========================================================
# DLA — LOOKAHEAD
# =========================================================

def crear_lookahead_scenarios(
    retornos,
    n_escenarios=200,
    horizonte=12,
    seed=777
):
    """
    Escenarios internos utilizados por la política DLA.

    OJO:
    No son los mismos escenarios de evaluación final.
    """

    return generar_escenarios(
        retornos,
        n_escenarios=n_escenarios,
        horizonte=horizonte,
        seed=seed
    )


def elegir_riesgo_dla(
    riqueza,
    pesos_actuales,
    pesos_base,
    escenarios_lookahead,
    perfil,
    riqueza_referencia,
    costo_transaccion=0.001,
    escalas=(0.50, 0.75, 1.00)
):
    """
    Direct Lookahead Approximation.

    La política compara diferentes niveles de exposición
    al portafolio recomendado.

    0.50 = 50% cartera + 50% cash
    0.75 = 75% cartera + 25% cash
    1.00 = 100% cartera
    """

    mejor_valor = -np.inf
    mejor_peso = pesos_base.copy()

    for escala in escalas:

        candidato = (
            escala
            * pesos_base
        )

        riqueza_trade, _, _ = (
            aplicar_trade(
                riqueza,
                pesos_actuales,
                candidato,
                costo_transaccion
            )
        )

        # Rendimientos mensuales del portafolio
        # para cada escenario interno.
        retornos_portafolio = (
            escenarios_lookahead
            @ candidato
        )

        factores = (
            1
            + retornos_portafolio
        )

        riqueza_terminal = (
            riqueza_trade
            * np.prod(
                factores,
                axis=1
            )
        )

        # Rendimiento relativo al punto de referencia
        diferencia = (
            riqueza_terminal
            / riqueza_referencia
            - 1
        )

        utilidad = prospect_value(
            diferencia,
            perfil["alpha"],
            perfil["beta"],
            perfil["lambda"]
        )

        valor_esperado = (
            np.mean(utilidad)
        )

        if valor_esperado > mejor_valor:

            mejor_valor = (
                valor_esperado
            )

            mejor_peso = (
                candidato.copy()
            )

    return mejor_peso


# =========================================================
# SIMULAR UNA TRAYECTORIA
# =========================================================

def simular_trayectoria(
    escenario,
    pesos_base,
    perfil,
    politica,
    capital_inicial=1_000_000,
    banda=0.08,
    panic_threshold=0.20,
    recovery_threshold=0.10,
    chasing_tilt=0.20,
    costo_transaccion=0.001,
    escenarios_lookahead=None,
    retornos_historicos=None,
    catalogo=None,
    ventana_cfa=60,
    frecuencia_cfa=12,
    max_activos=8
):
    """
    Simula una trayectoria de 120 meses.

    Estado:
        riqueza
        pesos actuales
        peak wealth
        retornos recientes

    Decisión:
        pesos objetivo / trades del mes

    Exogenous information:
        retornos del mercado del siguiente mes
    """

    riqueza = float(
        capital_inicial
    )

    riqueza_referencia = float(
        capital_inicial
    )

    peak = riqueza

    pesos_actuales = (
        pesos_base.copy()
    )

    costos_totales = 0.0
    trades = 0

    max_drawdown = 0.0

    # Variables para panic selling
    en_cash = False
    benchmark = 1.0
    benchmark_salida = None

    historia = []

    # Riqueza al inicio de cada mes (para el fan chart)
    trayectoria = [riqueza]

    for t in range(
        len(escenario)
    ):

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        drawdown = (
            0.0
            if peak <= 0
            else (
                peak - riqueza
            ) / peak
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        pesos_objetivo = (
            pesos_actuales.copy()
        )


        # =================================================
        # POLICY 1 — BUY & HOLD
        # =================================================

        if politica == "Buy & Hold":

            pesos_objetivo = (
                pesos_actuales.copy()
            )


        # =================================================
        # POLICY 2 — PFA BAND REBALANCING
        # =================================================

        elif politica == "Band Rebalancing (PFA)":

            desviacion = np.max(
                np.abs(
                    pesos_actuales
                    - pesos_base
                )
            )

            if desviacion >= banda:

                pesos_objetivo = (
                    pesos_base.copy()
                )


        # =================================================
        # POLICY 3 — DLA
        # =================================================

        elif politica == "Lookahead (DLA)":

            # Revisión anual
            if (
                t == 0
                or t % 12 == 0
            ):

                pesos_objetivo = (
                    elegir_riesgo_dla(
                        riqueza=riqueza,
                        pesos_actuales=pesos_actuales,
                        pesos_base=pesos_base,
                        escenarios_lookahead=escenarios_lookahead,
                        perfil=perfil,
                        riqueza_referencia=riqueza_referencia,
                        costo_transaccion=costo_transaccion
                    )
                )


        # =================================================
        # POLICY 4 — CFA: RE-OPTIMIZAR EL MILP CADA AÑO
        # =================================================

        elif politica == "Re-optimización anual (CFA)":

            # Cada "frecuencia_cfa" meses se vuelve a resolver
            # el MILP de la Parte 2 con los últimos "ventana_cfa"
            # meses disponibles: historia real + meses simulados.
            # Parámetros ajustables: ventana_cfa y frecuencia_cfa.
            if (
                t > 0
                and t % frecuencia_cfa == 0
            ):

                pasado = np.vstack(
                    [
                        retornos_historicos.to_numpy(),
                        np.array(historia)
                    ]
                )

                ventana = pd.DataFrame(
                    pasado[-ventana_cfa:],
                    columns=retornos_historicos.columns
                )

                try:

                    nuevo = optimizar_portafolio(
                        retornos=ventana,
                        catalogo=catalogo,
                        lambda_cliente=perfil["lambda"],
                        max_activos=max_activos
                    )

                    pesos_objetivo = alinear_pesos(
                        nuevo.pesos,
                        retornos_historicos.columns
                    )

                except Exception:

                    # Si Gurobi no encuentra solución,
                    # regresamos al portafolio original.
                    pesos_objetivo = (
                        pesos_base.copy()
                    )


        # =================================================
        # POLICY 5 — PANIC SELLING
        # =================================================

        elif politica == "Panic Selling":

            if not en_cash:

                if (
                    drawdown
                    >= panic_threshold
                ):

                    pesos_objetivo = (
                        np.zeros_like(
                            pesos_base
                        )
                    )

                    en_cash = True

                    benchmark_salida = (
                        benchmark
                    )

            else:

                if (
                    benchmark_salida
                    is not None
                    and benchmark
                    >= benchmark_salida
                    * (
                        1
                        + recovery_threshold
                    )
                ):

                    pesos_objetivo = (
                        pesos_base.copy()
                    )

                    en_cash = False

                    benchmark_salida = None

                    # Al volver a entrar reiniciamos el máximo.
                    # Sin esto el drawdown seguía arriba del
                    # umbral y el cliente volvía a vender al
                    # mes siguiente (entraba y salía cada mes).
                    peak = riqueza

                else:

                    pesos_objetivo = (
                        np.zeros_like(
                            pesos_base
                        )
                    )


        # =================================================
        # POLICY 6 — PERFORMANCE CHASING
        # =================================================

        elif politica == "Performance Chasing":

            if (
                t >= 12
                and t % 12 == 0
            ):

                recientes = np.array(
                    historia[-12:]
                )

                rendimiento_12m = (
                    np.prod(
                        1 + recientes,
                        axis=0
                    )
                    - 1
                )

                top = np.argsort(
                    rendimiento_12m
                )[-3:]

                ganadores = (
                    np.zeros_like(
                        pesos_base
                    )
                )

                ganadores[top] = (
                    1 / len(top)
                )

                pesos_objetivo = (
                    (1 - chasing_tilt)
                    * pesos_base
                    +
                    chasing_tilt
                    * ganadores
                )


        else:

            raise ValueError(
                f"Política desconocida: {politica}"
            )


        # -------------------------------------------------
        # DECISION / TRANSACTION
        # -------------------------------------------------

        if not np.allclose(
            pesos_objetivo,
            pesos_actuales
        ):

            (
                riqueza,
                costo_mes,
                turnover
            ) = aplicar_trade(
                riqueza,
                pesos_actuales,
                pesos_objetivo,
                costo_transaccion
            )

            costos_totales += (
                costo_mes
            )

            if turnover > 1e-8:
                trades += 1

            pesos_actuales = (
                pesos_objetivo.copy()
            )


        # -------------------------------------------------
        # EXOGENOUS INFORMATION
        # -------------------------------------------------

        retornos_mes = (
            escenario[t]
        )

        historia.append(
            retornos_mes.copy()
        )


        # -------------------------------------------------
        # TRANSITION
        # -------------------------------------------------

        rendimiento_portafolio = (
            np.dot(
                pesos_actuales,
                retornos_mes
            )
        )

        riqueza = (
            riqueza
            * (
                1
                + rendimiento_portafolio
            )
        )

        pesos_actuales = (
            actualizar_pesos(
                pesos_actuales,
                retornos_mes
            )
        )

        peak = max(
            peak,
            riqueza
        )

        trayectoria.append(
            riqueza
        )


        # Benchmark igual ponderado del mercado
        benchmark *= (
            1
            + np.mean(
                retornos_mes
            )
        )


    # =====================================================
    # OBJETIVO FINAL
    # =====================================================

    # Rendimiento total relativo al punto de referencia
    diferencia_terminal = (
        riqueza
        / riqueza_referencia
        - 1
    )

    valor = float(
        prospect_value(
            diferencia_terminal,
            perfil["alpha"],
            perfil["beta"],
            perfil["lambda"]
        )
    )


    return ResultadoTrayectoria(
        riqueza_final=riqueza,
        max_drawdown=max_drawdown,
        costos_transaccion=costos_totales,
        numero_trades=trades,
        valor_prospect=valor,
        trayectoria=np.array(
            trayectoria
        )
    )


# =========================================================
# TUNING DE PFA
# =========================================================

def tunear_banda(
    retornos,
    pesos_base,
    perfil,
    capital_inicial=1_000_000,
    bandas=(0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    n_escenarios=150,
    seed=111
):
    """
    Selecciona la banda usando escenarios distintos
    a los de evaluación final.
    """

    escenarios = generar_escenarios(
        retornos,
        n_escenarios=n_escenarios,
        horizonte=120,
        seed=seed
    )

    resultados = []

    for banda in bandas:

        valores = []

        for escenario in escenarios:

            r = simular_trayectoria(
                escenario=escenario,
                pesos_base=pesos_base,
                perfil=perfil,
                politica="Band Rebalancing (PFA)",
                capital_inicial=capital_inicial,
                banda=banda
            )

            valores.append(
                r.valor_prospect
            )

        resultados.append(
            {
                "banda": banda,
                "valor_prospect_medio": np.mean(
                    valores
                )
            }
        )

    tabla = pd.DataFrame(
        resultados
    )

    mejor = tabla.loc[
        tabla[
            "valor_prospect_medio"
        ].idxmax()
    ]

    return (
        float(
            mejor["banda"]
        ),
        tabla
    )


# =========================================================
# EVALUACIÓN DE TODAS LAS POLICIES
# =========================================================

def evaluar_politicas(
    retornos,
    pesos,
    perfil,
    capital_inicial=1_000_000,
    n_escenarios=500,
    seed_final=999,
    seed_tuning=111,
    catalogo=None,
    max_activos=8
):
    """
    Compara todas las políticas usando exactamente
    los mismos escenarios de mercado.
    """

    pesos_base = alinear_pesos(
        pesos,
        retornos.columns
    )

    # -----------------------------------------------------
    # TUNING
    # -----------------------------------------------------

    mejor_banda, tabla_tuning = (
        tunear_banda(
            retornos=retornos,
            pesos_base=pesos_base,
            perfil=perfil,
            capital_inicial=capital_inicial,
            seed=seed_tuning
        )
    )


    # -----------------------------------------------------
    # ESCENARIOS INTERNOS DE DLA
    # -----------------------------------------------------

    escenarios_lookahead = (
        crear_lookahead_scenarios(
            retornos,
            n_escenarios=200,
            horizonte=12,
            seed=777
        )
    )


    # -----------------------------------------------------
    # FINAL EVALUATION
    # -----------------------------------------------------

    escenarios = generar_escenarios(
        retornos,
        n_escenarios=n_escenarios,
        horizonte=120,
        seed=seed_final
    )


    politicas = [
        "Buy & Hold",
        "Band Rebalancing (PFA)",
        "Lookahead (DLA)",
        "Re-optimización anual (CFA)",
        "Panic Selling",
        "Performance Chasing"
    ]


    filas = []

    # Riqueza mes a mes de cada escenario, por política
    trayectorias = {}


    # Parámetros behavioral basados en cuestionario
    if perfil["panic_level"] == "Alto":
        panic_threshold = 0.15

    elif perfil["panic_level"] == "Moderado":
        panic_threshold = 0.20

    else:
        panic_threshold = 0.30


    if perfil["chasing_level"] == "Alto":
        chasing_tilt = 0.30

    elif perfil["chasing_level"] == "Moderado":
        chasing_tilt = 0.20

    else:
        chasing_tilt = 0.10


    for politica in politicas:

        caminos = []

        for n, escenario in enumerate(
            escenarios
        ):

            resultado = (
                simular_trayectoria(
                    escenario=escenario,
                    pesos_base=pesos_base,
                    perfil=perfil,
                    politica=politica,
                    capital_inicial=capital_inicial,
                    banda=mejor_banda,
                    panic_threshold=panic_threshold,
                    recovery_threshold=0.10,
                    chasing_tilt=chasing_tilt,
                    escenarios_lookahead=escenarios_lookahead,
                    retornos_historicos=retornos,
                    catalogo=catalogo,
                    max_activos=max_activos
                )
            )

            caminos.append(
                resultado.trayectoria
            )

            filas.append(
                {
                    "escenario": n,
                    "politica": politica,
                    "riqueza_final": resultado.riqueza_final,
                    "max_drawdown": resultado.max_drawdown,
                    "costos_transaccion": resultado.costos_transaccion,
                    "numero_trades": resultado.numero_trades,
                    "valor_prospect": resultado.valor_prospect
                }
            )

        trayectorias[politica] = np.array(
            caminos
        )


    resultados = pd.DataFrame(
        filas
    )


    return (
        resultados,
        mejor_banda,
        tabla_tuning,
        trayectorias
    )


# =========================================================
# RESUMEN ESTADÍSTICO
# =========================================================

def resumir_resultados(
    resultados,
    baseline="Buy & Hold"
):
    """
    Produce las métricas requeridas por el proyecto:
    media, desviación, percentiles e intervalo de confianza.

    La métrica principal es el valor prospect (el objetivo
    del modelo de Powell). También reportamos la diferencia
    contra el baseline escenario por escenario: como todas
    las políticas usan los mismos escenarios (common random
    numbers), el intervalo de la diferencia es mucho más
    angosto que comparar los intervalos por separado.
    """

    filas = []

    base = (
        resultados[
            resultados["politica"] == baseline
        ]
        .set_index("escenario")
    )

    for politica, df in resultados.groupby(
        "politica"
    ):

        riqueza = df[
            "riqueza_final"
        ]

        n = len(riqueza)

        media = (
            riqueza.mean()
        )

        sd = riqueza.std(
            ddof=1
        )

        error = (
            sd / np.sqrt(n)
        )

        ci_low = (
            media
            - 1.96 * error
        )

        ci_high = (
            media
            + 1.96 * error
        )

        # ---- Valor prospect (métrica principal) ----

        pv = df["valor_prospect"]

        pv_media = pv.mean()

        pv_sd = pv.std(
            ddof=1
        )

        pv_error = (
            pv_sd / np.sqrt(n)
        )

        # ---- Diferencia pareada contra el baseline ----

        actual = df.set_index("escenario")

        dif_pv = (
            actual["valor_prospect"]
            - base["valor_prospect"]
        )

        dif_riqueza = (
            actual["riqueza_final"]
            - base["riqueza_final"]
        )

        dif_pv_error = (
            dif_pv.std(ddof=1)
            / np.sqrt(n)
        )

        dif_riqueza_error = (
            dif_riqueza.std(ddof=1)
            / np.sqrt(n)
        )

        filas.append(
            {
                "politica": politica,

                "riqueza_media": media,

                "riqueza_mediana":
                    riqueza.median(),

                "percentil_5":
                    riqueza.quantile(0.05),

                "percentil_95":
                    riqueza.quantile(0.95),

                "desviacion":
                    sd,

                "IC95_low":
                    ci_low,

                "IC95_high":
                    ci_high,

                "drawdown_medio":
                    df[
                        "max_drawdown"
                    ].mean(),

                "costos_medios":
                    df[
                        "costos_transaccion"
                    ].mean(),

                "trades_medios":
                    df[
                        "numero_trades"
                    ].mean(),

                "prospect_value_medio":
                    pv_media,

                "prospect_value_sd":
                    pv_sd,

                "prospect_IC95_low":
                    pv_media - 1.96 * pv_error,

                "prospect_IC95_high":
                    pv_media + 1.96 * pv_error,

                "dif_prospect_vs_base":
                    dif_pv.mean(),

                "dif_prospect_IC95_low":
                    dif_pv.mean() - 1.96 * dif_pv_error,

                "dif_prospect_IC95_high":
                    dif_pv.mean() + 1.96 * dif_pv_error,

                "dif_riqueza_vs_base":
                    dif_riqueza.mean(),

                "dif_riqueza_IC95_low":
                    dif_riqueza.mean() - 1.96 * dif_riqueza_error,

                "dif_riqueza_IC95_high":
                    dif_riqueza.mean() + 1.96 * dif_riqueza_error
            }
        )


    resumen = pd.DataFrame(
        filas
    )

    return resumen.sort_values(
        "prospect_value_medio",
        ascending=False
    ).reset_index(
        drop=True
    )

if __name__ == "__main__":

    from datos import (
        cargar_catalogo,
        preparar_datos
    )

    from motor import (
        optimizar_portafolio
    )


    print(
        "Cargando datos..."
    )

    catalogo = (
        cargar_catalogo()
    )

    precios, retornos = (
        preparar_datos()
    )


    # Cliente de prueba
    perfil_prueba = {
        "lambda": 2.5,
        "alpha": 0.82,
        "beta": 0.75,
        "panic_level": "Moderado",
        "chasing_level": "Moderado"
    }


    print(
        "Construyendo portafolio..."
    )

    portafolio = (
        optimizar_portafolio(
            retornos=retornos,
            catalogo=catalogo,
            lambda_cliente=perfil_prueba[
                "lambda"
            ]
        )
    )


    print(
        "Simulando 500 escenarios..."
    )

    (
        resultados,
        banda,
        tuning,
        trayectorias
    ) = evaluar_politicas(
        retornos=retornos,
        pesos=portafolio.pesos,
        perfil=perfil_prueba,
        capital_inicial=1_000_000,
        n_escenarios=500,
        catalogo=catalogo
    )


    resumen = resumir_resultados(
        resultados
    )


    print(
        "\nBanda PFA seleccionada:",
        banda
    )


    print(
        "\nTUNING"
    )

    print(
        tuning
    )


    print(
        "\nRESULTADOS FINALES"
    )

    print(
        resumen[
            [
                "politica",
                "riqueza_media",
                "riqueza_mediana",
                "percentil_5",
                "percentil_95",
                "drawdown_medio",
                "prospect_value_medio",
                "prospect_IC95_low",
                "prospect_IC95_high"
            ]
        ].to_string(
            index=False
        )
    )