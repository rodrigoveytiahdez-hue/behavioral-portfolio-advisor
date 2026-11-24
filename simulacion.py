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

    # Diagnóstico de la política CFA
    cfa_fallos: int = 0
    cfa_ultimo_error: str = ""


# =========================================================
# PROSPECT THEORY
# =========================================================

def prospect_value(x, alpha, beta, lam):
    """
    Función de valor de Prospect Theory.

    x > 0 -> ganancias
    x < 0 -> pérdidas

    x se mide como rendimiento relativo al punto de referencia
    (por ejemplo, 0.25 = +25%), no en pesos.
    """
    x = np.asarray(x, dtype=float)

    return np.where(
        x >= 0,
        np.power(np.maximum(x, 0), alpha),
        -lam * np.power(np.maximum(-x, 0), beta),
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
    seed=2026,
):
    """
    Genera escenarios de rendimientos mensuales mediante block bootstrap.
    Los bloques contienen entre 3 y 6 meses consecutivos para conservar
    parte de la estructura temporal observada.
    """
    if retornos.empty:
        raise ValueError("No hay retornos para generar escenarios.")

    rng = np.random.default_rng(seed)
    datos = retornos.to_numpy(dtype=float)
    n_meses, n_activos = datos.shape

    if n_meses < bloque_min:
        raise ValueError("No hay suficientes meses para el block bootstrap.")

    bloque_max = min(bloque_max, n_meses)

    escenarios = np.zeros(
        (n_escenarios, horizonte, n_activos),
        dtype=float,
    )

    for n in range(n_escenarios):
        meses = []

        while len(meses) < horizonte:
            longitud = int(
                rng.integers(
                    bloque_min,
                    bloque_max + 1,
                )
            )

            max_inicio = n_meses - longitud
            inicio = int(
                rng.integers(
                    0,
                    max_inicio + 1,
                )
            )

            bloque = datos[
                inicio:inicio + longitud
            ]

            meses.extend(
                bloque.tolist()
            )

        escenarios[n] = np.array(
            meses[:horizonte],
            dtype=float,
        )

    return escenarios


# =========================================================
# PESOS
# =========================================================

def alinear_pesos(
    pesos,
    columnas,
):
    """
    Convierte la cartera seleccionada por Gurobi en un vector compatible
    con todos los activos de la matriz de retornos.
    """
    return np.array(
        [
            float(
                pesos.get(
                    ticker,
                    0.0,
                )
            )
            for ticker in columnas
        ],
        dtype=float,
    )


# =========================================================
# TRANSACCIONES
# =========================================================

def aplicar_trade(
    riqueza,
    pesos_actuales,
    pesos_objetivo,
    costo=0.001,
):
    """
    Costo de transacción = 10 basis points sobre el valor negociado.

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
        turnover,
    )


def actualizar_pesos(
    pesos,
    retornos_mes,
):
    """
    Actualiza los pesos después de los rendimientos del mes.

    Si los pesos suman menos de 1, la diferencia se interpreta
    como cash con rendimiento 0%.
    """

    crecimiento = (
        pesos
        * (1 + retornos_mes)
    )

    rendimiento_portafolio = np.dot(
        pesos,
        retornos_mes,
    )

    denominador = (
        1
        + rendimiento_portafolio
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
# DLA — DIRECT LOOKAHEAD APPROXIMATION
# =========================================================

def crear_lookahead_scenarios(
    retornos,
    n_escenarios=200,
    horizonte=12,
    seed=777,
):
    """
    Escenarios internos utilizados por la política DLA.

    No son los mismos escenarios de la evaluación final.
    """

    return generar_escenarios(
        retornos=retornos,
        n_escenarios=n_escenarios,
        horizonte=horizonte,
        seed=seed,
    )


def elegir_riesgo_dla(
    riqueza,
    pesos_actuales,
    pesos_base,
    escenarios_lookahead,
    perfil,
    riqueza_referencia,
    costo_transaccion=0.001,
    escalas=(0.50, 0.75, 1.00),
):
    """
    Direct Lookahead Approximation.

    Compara diferentes niveles de exposición:

    0.50 = 50% cartera + 50% cash
    0.75 = 75% cartera + 25% cash
    1.00 = 100% cartera
    """

    if escenarios_lookahead is None:
        raise ValueError(
            "DLA necesita escenarios de lookahead."
        )

    mejor_valor = -np.inf
    mejor_peso = pesos_base.copy()

    for escala in escalas:

        candidato = (
            escala
            * pesos_base
        )

        (
            riqueza_trade,
            _,
            _
        ) = aplicar_trade(
            riqueza=riqueza,
            pesos_actuales=pesos_actuales,
            pesos_objetivo=candidato,
            costo=costo_transaccion,
        )

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
                axis=1,
            )
        )

        diferencia = (
            riqueza_terminal
            / riqueza_referencia
            - 1
        )

        utilidad = prospect_value(
            diferencia,
            perfil["alpha"],
            perfil["beta"],
            perfil["lambda"],
        )

        valor_esperado = float(
            np.mean(
                utilidad
            )
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
    max_activos=8,
):
    """
    Simula una trayectoria mensual.

    Estado:
        - riqueza
        - pesos actuales
        - máximo histórico de riqueza
        - retornos recientes

    Decisión:
        - pesos objetivo / trades del mes

    Información exógena:
        - retornos del mercado del siguiente mes
    """

    riqueza = float(
        capital_inicial
    )

    riqueza_referencia = float(
        capital_inicial
    )

    peak = riqueza

    pesos_base = np.asarray(
        pesos_base,
        dtype=float,
    )

    pesos_actuales = (
        pesos_base.copy()
    )

    costos_totales = 0.0
    trades = 0
    max_drawdown = 0.0

    # Diagnóstico de CFA
    cfa_fallos = 0
    cfa_ultimo_error = ""

    # Variables para panic selling
    en_cash = False
    benchmark = 1.0
    benchmark_salida = None

    historia = []

    # Para el fan chart
    trayectoria = [
        riqueza
    ]

    for t in range(
        len(escenario)
    ):

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        drawdown = (
            0.0
            if peak <= 0
            else max(
                0.0,
                (
                    peak
                    - riqueza
                )
                / peak
            )
        )

        max_drawdown = max(
            max_drawdown,
            drawdown,
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
                        costo_transaccion=costo_transaccion,
                    )
                )


        # =================================================
        # POLICY 4 — CFA
        # =================================================

        elif politica == "Re-optimización anual (CFA)":

            if (
                t > 0
                and t % frecuencia_cfa == 0
            ):

                try:

                    if retornos_historicos is None:

                        raise ValueError(
                            "CFA necesita retornos_historicos."
                        )

                    if catalogo is None:

                        raise ValueError(
                            "CFA necesita el catálogo de activos."
                        )

                    pasado = np.vstack(
                        [
                            retornos_historicos.to_numpy(),
                            np.array(
                                historia
                            ),
                        ]
                    )

                    ventana = pd.DataFrame(
                        pasado[
                            -ventana_cfa:
                        ],
                        columns=retornos_historicos.columns,
                    )

                    nuevo = optimizar_portafolio(
                        retornos=ventana,
                        catalogo=catalogo,
                        lambda_cliente=perfil[
                            "lambda"
                        ],
                        max_activos=max_activos,
                    )

                    pesos_objetivo = (
                        alinear_pesos(
                            nuevo.pesos,
                            retornos_historicos.columns,
                        )
                    )

                except Exception as exc:

                    cfa_fallos += 1

                    cfa_ultimo_error = (
                        str(
                            exc
                        )
                    )

                    # Si CFA falla, seguimos con
                    # el portafolio base.
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

                    # Reiniciamos el máximo para evitar
                    # que vuelva a vender inmediatamente.
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
                    historia[
                        -12:
                    ]
                )

                rendimiento_12m = (
                    np.prod(
                        1
                        + recientes,
                        axis=0,
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

                ganadores[
                    top
                ] = (
                    1
                    / len(
                        top
                    )
                )

                pesos_objetivo = (
                    (
                        1
                        - chasing_tilt
                    )
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
            pesos_actuales,
        ):

            (
                riqueza,
                costo_mes,
                turnover,
            ) = aplicar_trade(
                riqueza=riqueza,
                pesos_actuales=pesos_actuales,
                pesos_objetivo=pesos_objetivo,
                costo=costo_transaccion,
            )

            costos_totales += (
                costo_mes
            )

            if turnover > 1e-8:

                trades += 1

            pesos_actuales = (
                pesos_objetivo.copy()
            )

            drawdown_post_trade = (
                0.0
                if peak <= 0
                else max(
                    0.0,
                    (
                        peak
                        - riqueza
                    )
                    / peak
                )
            )

            max_drawdown = max(
                max_drawdown,
                drawdown_post_trade,
            )


        # -------------------------------------------------
        # EXOGENOUS INFORMATION
        # -------------------------------------------------

        retornos_mes = np.asarray(
            escenario[t],
            dtype=float,
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
                retornos_mes,
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
                retornos_mes,
            )
        )

        drawdown_fin_mes = (
            0.0
            if peak <= 0
            else max(
                0.0,
                (
                    peak
                    - riqueza
                )
                / peak
            )
        )

        max_drawdown = max(
            max_drawdown,
            drawdown_fin_mes,
        )

        peak = max(
            peak,
            riqueza,
        )

        trayectoria.append(
            riqueza
        )


        # Benchmark igual ponderado
        benchmark *= (
            1
            + np.mean(
                retornos_mes
            )
        )


    # =====================================================
    # OBJETIVO FINAL
    # =====================================================

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
            perfil["lambda"],
        )
    )

    return ResultadoTrayectoria(
        riqueza_final=riqueza,
        max_drawdown=max_drawdown,
        costos_transaccion=costos_totales,
        numero_trades=trades,
        valor_prospect=valor,
        trayectoria=np.array(
            trayectoria,
            dtype=float,
        ),
        cfa_fallos=cfa_fallos,
        cfa_ultimo_error=cfa_ultimo_error,
    )


# =========================================================
# TUNING DE PFA
# =========================================================

def tunear_banda(
    retornos,
    pesos_base,
    perfil,
    capital_inicial=1_000_000,
    bandas=(
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    ),
    n_escenarios=150,
    seed=111,
):
    """
    Selecciona la banda usando escenarios distintos
    a los de evaluación final.
    """

    escenarios = generar_escenarios(
        retornos=retornos,
        n_escenarios=n_escenarios,
        horizonte=120,
        seed=seed,
    )

    resultados = []

    for banda in bandas:

        valores = []

        for escenario in escenarios:

            resultado = simular_trayectoria(
                escenario=escenario,
                pesos_base=pesos_base,
                perfil=perfil,
                politica="Band Rebalancing (PFA)",
                capital_inicial=capital_inicial,
                banda=banda,
            )

            valores.append(
                resultado.valor_prospect
            )

        resultados.append(
            {
                "banda": banda,
                "valor_prospect_medio": float(
                    np.mean(
                        valores
                    )
                ),
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
            mejor[
                "banda"
            ]
        ),
        tabla,
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
    max_activos=8,
):
    """
    Compara todas las políticas usando exactamente
    los mismos escenarios.

    Retorna:
        resultados
        mejor_banda
        tabla_tuning
        trayectorias
    """

    pesos_base = alinear_pesos(
        pesos,
        retornos.columns,
    )


    # -----------------------------------------------------
    # TUNING PFA
    # -----------------------------------------------------

    (
        mejor_banda,
        tabla_tuning,
    ) = tunear_banda(
        retornos=retornos,
        pesos_base=pesos_base,
        perfil=perfil,
        capital_inicial=capital_inicial,
        seed=seed_tuning,
    )


    # -----------------------------------------------------
    # ESCENARIOS INTERNOS DE DLA
    # -----------------------------------------------------

    escenarios_lookahead = (
        crear_lookahead_scenarios(
            retornos=retornos,
            n_escenarios=200,
            horizonte=12,
            seed=777,
        )
    )


    # -----------------------------------------------------
    # ESCENARIOS DE EVALUACIÓN FINAL
    # -----------------------------------------------------

    escenarios = generar_escenarios(
        retornos=retornos,
        n_escenarios=n_escenarios,
        horizonte=120,
        seed=seed_final,
    )


    politicas = [
        "Buy & Hold",
        "Band Rebalancing (PFA)",
        "Lookahead (DLA)",
        "Re-optimización anual (CFA)",
        "Panic Selling",
        "Performance Chasing",
    ]


    filas = []

    trayectorias = {}


    # -----------------------------------------------------
    # PARÁMETROS BEHAVIORAL
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MISMO ESCENARIO PARA TODAS LAS POLICIES
    # -----------------------------------------------------

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
                    max_activos=max_activos,
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
                    "valor_prospect": resultado.valor_prospect,
                    "cfa_fallos": resultado.cfa_fallos,
                    "cfa_ultimo_error": resultado.cfa_ultimo_error,
                }
            )

        trayectorias[
            politica
        ] = np.array(
            caminos,
            dtype=float,
        )


    resultados = pd.DataFrame(
        filas
    )


    return (
        resultados,
        mejor_banda,
        tabla_tuning,
        trayectorias,
    )


# =========================================================
# RESUMEN ESTADÍSTICO
# =========================================================

def resumir_resultados(
    resultados,
    baseline="Buy & Hold",
):
    """
    Produce las métricas requeridas:

    - media
    - mediana
    - desviación estándar
    - percentiles
    - intervalo de confianza
    - drawdown
    - costos
    - trades
    - Prospect Value

    También calcula diferencias pareadas contra Buy & Hold.
    """

    if resultados.empty:

        raise ValueError(
            "No hay resultados para resumir."
        )


    base = (
        resultados[
            resultados[
                "politica"
            ] == baseline
        ]
        .set_index(
            "escenario"
        )
        .sort_index()
    )


    if base.empty:

        raise ValueError(
            f"No se encontró el baseline '{baseline}'."
        )


    filas = []


    for politica, df in resultados.groupby(
        "politica"
    ):

        df = df.sort_values(
            "escenario"
        ).copy()

        riqueza = df[
            "riqueza_final"
        ]

        n = len(
            riqueza
        )


        # -------------------------------------------------
        # RIQUEZA
        # -------------------------------------------------

        media = float(
            riqueza.mean()
        )

        sd = float(
            riqueza.std(
                ddof=1
            )
        )

        error = (
            sd
            / np.sqrt(
                n
            )
        )

        ci_low = (
            media
            - 1.96
            * error
        )

        ci_high = (
            media
            + 1.96
            * error
        )


        # -------------------------------------------------
        # PROSPECT VALUE
        # -------------------------------------------------

        pv = df[
            "valor_prospect"
        ]

        pv_media = float(
            pv.mean()
        )

        pv_sd = float(
            pv.std(
                ddof=1
            )
        )

        pv_error = (
            pv_sd
            / np.sqrt(
                n
            )
        )


        # -------------------------------------------------
        # DIFERENCIAS PAREADAS VS BASELINE
        # -------------------------------------------------

        actual = df.set_index(
            "escenario"
        )

        comunes = actual.index.intersection(
            base.index
        )

        actual_comun = actual.loc[
            comunes
        ]

        base_comun = base.loc[
            comunes
        ]


        dif_pv = (
            actual_comun[
                "valor_prospect"
            ]
            - base_comun[
                "valor_prospect"
            ]
        )


        dif_riqueza = (
            actual_comun[
                "riqueza_final"
            ]
            - base_comun[
                "riqueza_final"
            ]
        )


        n_dif = len(
            comunes
        )


        if n_dif > 1:

            dif_pv_error = (
                dif_pv.std(
                    ddof=1
                )
                / np.sqrt(
                    n_dif
                )
            )

            dif_riqueza_error = (
                dif_riqueza.std(
                    ddof=1
                )
                / np.sqrt(
                    n_dif
                )
            )

        else:

            dif_pv_error = 0.0

            dif_riqueza_error = 0.0


        # -------------------------------------------------
        # DIAGNÓSTICO CFA
        # -------------------------------------------------

        if "cfa_fallos" in df.columns:

            cfa_fallos_total = int(
                df[
                    "cfa_fallos"
                ].sum()
            )

            cfa_escenarios_con_fallo = int(
                (
                    df[
                        "cfa_fallos"
                    ]
                    > 0
                ).sum()
            )

        else:

            cfa_fallos_total = 0
            cfa_escenarios_con_fallo = 0


        # -------------------------------------------------
        # FILA FINAL
        # -------------------------------------------------

        filas.append(
            {
                "politica": politica,

                "riqueza_media": media,

                "riqueza_mediana": float(
                    riqueza.median()
                ),

                "percentil_5": float(
                    riqueza.quantile(
                        0.05
                    )
                ),

                "percentil_95": float(
                    riqueza.quantile(
                        0.95
                    )
                ),

                "desviacion": sd,

                "IC95_low": ci_low,

                "IC95_high": ci_high,

                "drawdown_medio": float(
                    df[
                        "max_drawdown"
                    ].mean()
                ),

                "costos_medios": float(
                    df[
                        "costos_transaccion"
                    ].mean()
                ),

                "trades_medios": float(
                    df[
                        "numero_trades"
                    ].mean()
                ),

                "prospect_value_medio": pv_media,

                "prospect_value_sd": pv_sd,

                "prospect_IC95_low": (
                    pv_media
                    - 1.96
                    * pv_error
                ),

                "prospect_IC95_high": (
                    pv_media
                    + 1.96
                    * pv_error
                ),

                "dif_prospect_vs_base": float(
                    dif_pv.mean()
                ),

                "dif_prospect_IC95_low": float(
                    dif_pv.mean()
                    - 1.96
                    * dif_pv_error
                ),

                "dif_prospect_IC95_high": float(
                    dif_pv.mean()
                    + 1.96
                    * dif_pv_error
                ),

                "dif_riqueza_vs_base": float(
                    dif_riqueza.mean()
                ),

                "dif_riqueza_IC95_low": float(
                    dif_riqueza.mean()
                    - 1.96
                    * dif_riqueza_error
                ),

                "dif_riqueza_IC95_high": float(
                    dif_riqueza.mean()
                    + 1.96
                    * dif_riqueza_error
                ),

                "cfa_fallos_total": (
                    cfa_fallos_total
                ),

                "cfa_escenarios_con_fallo": (
                    cfa_escenarios_con_fallo
                ),
            }
        )


    resumen = pd.DataFrame(
        filas
    )


    return (
        resumen
        .sort_values(
            "prospect_value_medio",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# PRUEBA LOCAL
# =========================================================

if __name__ == "__main__":

    from datos import (
        cargar_catalogo,
        preparar_datos,
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


    perfil_prueba = {
        "lambda": 2.5,
        "alpha": 0.82,
        "beta": 0.75,
        "panic_level": "Moderado",
        "chasing_level": "Moderado",
    }


    print(
        "Construyendo portafolio..."
    )


    portafolio = optimizar_portafolio(
        retornos=retornos,
        catalogo=catalogo,
        lambda_cliente=perfil_prueba[
            "lambda"
        ],
    )


    print(
        "Simulando 500 escenarios..."
    )


    (
        resultados,
        banda,
        tuning,
        trayectorias,
    ) = evaluar_politicas(
        retornos=retornos,
        pesos=portafolio.pesos,
        perfil=perfil_prueba,
        capital_inicial=1_000_000,
        n_escenarios=500,
        catalogo=catalogo,
    )


    resumen = resumir_resultados(
        resultados
    )


    print(
        "\nBanda PFA seleccionada:",
        banda,
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
                "prospect_IC95_high",
                "cfa_fallos_total",
                "cfa_escenarios_con_fallo",
            ]
        ].to_string(
            index=False
        )
    )