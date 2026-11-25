from dataclasses import dataclass

import numpy as np
import pandas as pd

from motor import optimizar_portafolio


@dataclass
class ResultadoTrayectoria:
    riqueza_final: float
    max_drawdown: float
    costos_transaccion: float
    numero_trades: int
    valor_prospect: float
    trayectoria: np.ndarray

    cfa_fallos: int = 0
    cfa_ultimo_error: str = ""


# =========================================================
# PROSPECT THEORY
# =========================================================

def prospect_value(
    x,
    alpha,
    beta,
    lam
):

    x = np.asarray(
        x,
        dtype=float
    )

    return np.where(
        x >= 0,
        np.power(
            np.maximum(
                x,
                0
            ),
            alpha
        ),
        -lam
        * np.power(
            np.maximum(
                -x,
                0
            ),
            beta
        )
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

    if retornos.empty:

        raise ValueError(
            "No hay retornos para generar escenarios."
        )

    rng = np.random.default_rng(
        seed
    )

    datos = retornos.to_numpy(
        dtype=float
    )

    n_meses, n_activos = (
        datos.shape
    )

    bloque_max = min(
        bloque_max,
        n_meses
    )

    escenarios = np.zeros(
        (
            n_escenarios,
            horizonte,
            n_activos
        ),
        dtype=float
    )

    for n in range(
        n_escenarios
    ):

        meses = []

        while len(meses) < horizonte:

            longitud = int(
                rng.integers(
                    bloque_min,
                    bloque_max + 1
                )
            )

            inicio = int(
                rng.integers(
                    0,
                    n_meses
                    - longitud
                    + 1
                )
            )

            bloque = datos[
                inicio:
                inicio + longitud
            ]

            meses.extend(
                bloque.tolist()
            )

        escenarios[n] = (
            np.array(
                meses[:horizonte],
                dtype=float
            )
        )

    return escenarios


# =========================================================
# PESOS
# =========================================================

def alinear_pesos(
    pesos,
    columnas
):

    return np.array(
        [
            float(
                pesos.get(
                    ticker,
                    0.0
                )
            )
            for ticker in columnas
        ],
        dtype=float
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

    crecimiento = (
        pesos
        * (
            1
            + retornos_mes
        )
    )

    rendimiento = np.dot(
        pesos,
        retornos_mes
    )

    denominador = (
        1
        + rendimiento
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
# DLA
# =========================================================

def crear_lookahead_scenarios(
    retornos,
    n_escenarios=200,
    horizonte=12,
    seed=777
):

    return generar_escenarios(
        retornos=retornos,
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
    escalas=(
        0.50,
        0.75,
        1.00
    )
):

    mejor_valor = (
        -np.inf
    )

    mejor_peso = (
        pesos_base.copy()
    )

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
            riqueza,
            pesos_actuales,
            candidato,
            costo_transaccion
        )

        retornos_portafolio = (
            escenarios_lookahead
            @ candidato
        )

        riqueza_terminal = (
            riqueza_trade
            * np.prod(
                1
                + retornos_portafolio,
                axis=1
            )
        )

        diferencia = (
            riqueza_terminal
            / riqueza_referencia
            - 1
        )

        utilidad = (
            prospect_value(
                diferencia,
                perfil[
                    "alpha"
                ],
                perfil[
                    "beta"
                ],
                perfil[
                    "lambda"
                ]
            )
        )

        valor = float(
            np.mean(
                utilidad
            )
        )

        if valor > mejor_valor:

            mejor_valor = (
                valor
            )

            mejor_peso = (
                candidato.copy()
            )

    return mejor_peso


# =========================================================
# UNA TRAYECTORIA
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

    riqueza = float(
        capital_inicial
    )

    riqueza_referencia = float(
        capital_inicial
    )

    peak = riqueza

    pesos_base = np.asarray(
        pesos_base,
        dtype=float
    )

    pesos_actuales = (
        pesos_base.copy()
    )

    costos_totales = 0.0
    trades = 0
    max_drawdown = 0.0

    cfa_fallos = 0
    cfa_ultimo_error = ""

    en_cash = False
    benchmark = 1.0
    benchmark_salida = None

    historia = []

    trayectoria = [
        riqueza
    ]


    for t in range(
        len(escenario)
    ):

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
            drawdown
        )

        pesos_objetivo = (
            pesos_actuales.copy()
        )


        # =================================================
        # BUY & HOLD
        # =================================================

        if politica == "Buy & Hold":

            pass


        # =================================================
        # PFA
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
        # DLA
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
                        costo_transaccion=costo_transaccion
                    )
                )


        # =================================================
        # CFA
        # =================================================

        elif politica == "Re-optimización anual (CFA)":

            if (
                t > 0
                and t
                % frecuencia_cfa
                == 0
            ):

                try:

                    if retornos_historicos is None:

                        raise ValueError(
                            "CFA necesita retornos históricos."
                        )

                    if catalogo is None:

                        raise ValueError(
                            "CFA necesita el catálogo."
                        )


                    pasado = np.vstack(
                        [
                            retornos_historicos.to_numpy(),
                            np.array(
                                historia
                            )
                        ]
                    )

                    ventana = pd.DataFrame(
                        pasado[
                            -ventana_cfa:
                        ],
                        columns=retornos_historicos.columns
                    )


                    nuevo = (
                        optimizar_portafolio(
                            retornos=ventana,
                            catalogo=catalogo,
                            lambda_cliente=perfil[
                                "lambda"
                            ],
                            perfil_cliente=perfil.get(
                                "perfil",
                                "Moderado"
                            ),
                            max_activos=max_activos
                        )
                    )


                    pesos_objetivo = (
                        alinear_pesos(
                            nuevo.pesos,
                            retornos_historicos.columns
                        )
                    )

                except Exception as exc:

                    cfa_fallos += 1

                    cfa_ultimo_error = (
                        str(
                            exc
                        )
                    )

                    pesos_objetivo = (
                        pesos_base.copy()
                    )


        # =================================================
        # PANIC SELLING
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

                    peak = riqueza

                else:

                    pesos_objetivo = (
                        np.zeros_like(
                            pesos_base
                        )
                    )


        # =================================================
        # PERFORMANCE CHASING
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


        # =================================================
        # TRADE
        # =================================================

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


        # =================================================
        # INFORMACIÓN EXÓGENA
        # =================================================

        retornos_mes = np.asarray(
            escenario[t],
            dtype=float
        )

        historia.append(
            retornos_mes.copy()
        )


        # =================================================
        # TRANSICIÓN
        # =================================================

        rendimiento = np.dot(
            pesos_actuales,
            retornos_mes
        )

        riqueza *= (
            1
            + rendimiento
        )

        pesos_actuales = (
            actualizar_pesos(
                pesos_actuales,
                retornos_mes
            )
        )


        drawdown_fin = (
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
            drawdown_fin
        )

        peak = max(
            peak,
            riqueza
        )

        trayectoria.append(
            riqueza
        )


        benchmark *= (
            1
            + np.mean(
                retornos_mes
            )
        )


    diferencia_terminal = (
        riqueza
        / riqueza_referencia
        - 1
    )


    valor = float(
        prospect_value(
            diferencia_terminal,
            perfil[
                "alpha"
            ],
            perfil[
                "beta"
            ],
            perfil[
                "lambda"
            ]
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
        ),
        cfa_fallos=cfa_fallos,
        cfa_ultimo_error=cfa_ultimo_error
    )


# =========================================================
# TUNING PFA
# =========================================================

def tunear_banda(
    retornos,
    pesos_base,
    perfil,
    capital_inicial=1_000_000,
    bandas=(
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ),
    n_escenarios=150,
    seed=111
):

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

            resultado = (
                simular_trayectoria(
                    escenario=escenario,
                    pesos_base=pesos_base,
                    perfil=perfil,
                    politica="Band Rebalancing (PFA)",
                    capital_inicial=capital_inicial,
                    banda=banda
                )
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
            mejor[
                "banda"
            ]
        ),
        tabla
    )


# =========================================================
# EVALUACIÓN
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

    pesos_base = (
        alinear_pesos(
            pesos,
            retornos.columns
        )
    )


    (
        mejor_banda,
        tabla_tuning
    ) = tunear_banda(
        retornos=retornos,
        pesos_base=pesos_base,
        perfil=perfil,
        capital_inicial=capital_inicial,
        seed=seed_tuning
    )


    escenarios_lookahead = (
        crear_lookahead_scenarios(
            retornos,
            n_escenarios=200,
            horizonte=12,
            seed=777
        )
    )


    escenarios = (
        generar_escenarios(
            retornos,
            n_escenarios=n_escenarios,
            horizonte=120,
            seed=seed_final
        )
    )


    politicas = [
        "Buy & Hold",
        "Band Rebalancing (PFA)",
        "Lookahead (DLA)",
        "Re-optimización anual (CFA)",
        "Panic Selling",
        "Performance Chasing"
    ]


    if perfil[
        "panic_level"
    ] == "Alto":

        panic_threshold = 0.15

    elif perfil[
        "panic_level"
    ] == "Moderado":

        panic_threshold = 0.20

    else:

        panic_threshold = 0.30


    if perfil[
        "chasing_level"
    ] == "Alto":

        chasing_tilt = 0.30

    elif perfil[
        "chasing_level"
    ] == "Moderado":

        chasing_tilt = 0.20

    else:

        chasing_tilt = 0.10


    filas = []

    trayectorias = {}


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
                    "valor_prospect": resultado.valor_prospect,
                    "cfa_fallos": resultado.cfa_fallos,
                    "cfa_ultimo_error": resultado.cfa_ultimo_error
                }
            )


        trayectorias[
            politica
        ] = np.array(
            caminos
        )


    return (
        pd.DataFrame(
            filas
        ),
        mejor_banda,
        tabla_tuning,
        trayectorias
    )


# =========================================================
# RESUMEN
# =========================================================

def resumir_resultados(
    resultados,
    baseline="Buy & Hold"
):

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


    filas = []


    for politica, df in (
        resultados.groupby(
            "politica"
        )
    ):

        df = (
            df
            .sort_values(
                "escenario"
            )
            .copy()
        )


        riqueza = df[
            "riqueza_final"
        ]

        n = len(
            riqueza
        )

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


        actual = (
            df
            .set_index(
                "escenario"
            )
        )

        comunes = (
            actual.index
            .intersection(
                base.index
            )
        )


        dif_pv = (
            actual.loc[
                comunes,
                "valor_prospect"
            ]
            -
            base.loc[
                comunes,
                "valor_prospect"
            ]
        )


        dif_riqueza = (
            actual.loc[
                comunes,
                "riqueza_final"
            ]
            -
            base.loc[
                comunes,
                "riqueza_final"
            ]
        )


        if len(
            dif_pv
        ) > 1:

            error_dif_pv = (
                dif_pv.std(
                    ddof=1
                )
                / np.sqrt(
                    len(
                        dif_pv
                    )
                )
            )

            error_dif_riqueza = (
                dif_riqueza.std(
                    ddof=1
                )
                / np.sqrt(
                    len(
                        dif_riqueza
                    )
                )
            )

        else:

            error_dif_pv = 0.0
            error_dif_riqueza = 0.0


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

                "IC95_low": (
                    media
                    - 1.96
                    * error
                ),

                "IC95_high": (
                    media
                    + 1.96
                    * error
                ),

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

                "prospect_value_medio": (
                    pv_media
                ),

                "prospect_value_sd": (
                    pv_sd
                ),

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
                    * error_dif_pv
                ),

                "dif_prospect_IC95_high": float(
                    dif_pv.mean()
                    + 1.96
                    * error_dif_pv
                ),

                "dif_riqueza_vs_base": float(
                    dif_riqueza.mean()
                ),

                "dif_riqueza_IC95_low": float(
                    dif_riqueza.mean()
                    - 1.96
                    * error_dif_riqueza
                ),

                "dif_riqueza_IC95_high": float(
                    dif_riqueza.mean()
                    + 1.96
                    * error_dif_riqueza
                ),

                "cfa_fallos_total": int(
                    df[
                        "cfa_fallos"
                    ].sum()
                ),

                "cfa_escenarios_con_fallo": int(
                    (
                        df[
                            "cfa_fallos"
                        ] > 0
                    ).sum()
                )
            }
        )


    return (
        pd.DataFrame(
            filas
        )
        .sort_values(
            "prospect_value_medio",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )