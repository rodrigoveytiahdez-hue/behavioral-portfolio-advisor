from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


@dataclass
class ResultadoPortafolio:
    pesos: pd.Series
    rendimiento_mensual: float
    rendimiento_anual: float
    cvar_mensual: float
    lambda_cliente: float
    penalizacion_riesgo: float
    numero_activos: int
    estado: str


# =========================================================
# PERFIL Y RIESGO
# =========================================================

def penalizacion_desde_lambda(lam):
    """
    Lambda controla cuánto penalizamos el CVaR.

    Lambda bajo  -> mayor tolerancia a pérdidas.
    Lambda alto  -> mayor sensibilidad a pérdidas.
    """

    lam = float(lam)

    lam = np.clip(
        lam,
        1.5,
        4.5
    )

    minimo = 0.05
    maximo = 0.50

    normalizado = (
        (lam - 1.5)
        / (4.5 - 1.5)
    )

    return (
        minimo
        + normalizado
        * (maximo - minimo)
    )


def limites_desde_perfil(perfil):
    """
    Restricciones de suitability.

    El perfil controla límites de asignación.
    Lambda sigue controlando la penalización por riesgo.
    """

    perfil = (
        str(perfil)
        .strip()
        .lower()
    )

    if perfil == "conservador":

        return {
            "min_bonos": 0.40,
            "max_equity": 0.45,
            "max_growth_tech": 0.25,
            "max_alternativos": 0.10,
            "peso_maximo": 0.20,
        }

    elif perfil == "moderado":

        return {
            "min_bonos": 0.20,
            "max_equity": 0.70,
            "max_growth_tech": 0.40,
            "max_alternativos": 0.15,
            "peso_maximo": 0.25,
        }

    # Crecimiento
    return {
        "min_bonos": 0.05,
        "max_equity": 0.90,
        "max_growth_tech": 0.60,
        "max_alternativos": 0.20,
        "peso_maximo": 0.30,
    }


# =========================================================
# VALIDACIÓN
# =========================================================

def validar_datos(
    retornos,
    catalogo
):
    if retornos.empty:

        raise ValueError(
            "No existen rendimientos para construir el portafolio."
        )

    if len(retornos) < 60:

        raise ValueError(
            "Se necesitan al menos 60 meses de información."
        )

    columnas_retornos = set(
        retornos.columns
    )

    comunes = [
        ticker
        for ticker in catalogo["ticker"]
        if ticker in columnas_retornos
    ]

    if len(comunes) < 10:

        raise ValueError(
            "No hay suficientes activos con información completa."
        )

    return comunes


def _texto(valor):

    return (
        str(valor)
        .strip()
        .lower()
    )


# =========================================================
# CLASIFICACIÓN DE ACTIVOS
# =========================================================

def _indices_bonos(
    tickers,
    clases
):

    tickers_bonos = {
        "BND",
        "AGG",
        "SHY",
        "IEF",
        "TLT",
        "TIP",
        "LQD",
        "HYG",
        "VGSH",
        "VGIT",
        "VGLT",
    }

    indices = []

    for i, ticker in enumerate(
        tickers
    ):

        clase = _texto(
            clases[i]
        )

        if (
            "bond" in clase
            or "fixed income" in clase
            or "renta fija" in clase
            or ticker.upper()
            in tickers_bonos
        ):

            indices.append(i)

    return indices


def _indices_equity(
    tickers,
    clases,
    sectores
):

    tickers_equity = {
        "VOO",
        "SPY",
        "IVV",
        "QQQ",
        "QQQM",
        "IWM",
        "IWF",
        "IWD",
        "VTI",
        "VEA",
        "EFA",
        "EEM",
        "VWO",
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLY",
        "XLP",
        "XLI",
        "XLC",
        "XLU",
        "XLRE",
    }

    indices = []

    for i, ticker in enumerate(
        tickers
    ):

        clase = _texto(
            clases[i]
        )

        sector = _texto(
            sectores[i]
        )

        if (
            "equity" in clase
            or "stock" in clase
            or "acciones" in clase
            or "sector" in clase
            or "equity" in sector
            or ticker.upper()
            in tickers_equity
        ):

            indices.append(i)

    return indices


def _indices_growth_tech(
    tickers,
    clases,
    sectores
):

    tickers_growth = {
        "QQQ",
        "QQQM",
        "IWF",
        "VUG",
        "XLK",
        "SOXX",
        "SMH",
    }

    indices = []

    for i, ticker in enumerate(
        tickers
    ):

        clase = _texto(
            clases[i]
        )

        sector = _texto(
            sectores[i]
        )

        if (
            "technology" in sector
            or "tech" in sector
            or "growth" in sector
            or "technology" in clase
            or "growth" in clase
            or ticker.upper()
            in tickers_growth
        ):

            indices.append(i)

    return indices


# =========================================================
# OPTIMIZACIÓN
# =========================================================

def optimizar_portafolio(
    retornos,
    catalogo,
    lambda_cliente,
    perfil_cliente="Moderado",
    max_activos=8,
    peso_minimo=0.05,
    peso_maximo=0.30,
    max_sector=0.40,
    max_clase=0.50,
    max_alternativos=0.20,
    alpha_cvar=0.95,
):
    """
    Portafolio Mean-CVaR con restricciones de suitability.

    Objetivo:
        maximizar rendimiento esperado
        - penalización * CVaR

    Lambda:
        controla la penalización por riesgo.

    Perfil:
        controla límites de bonos, equity,
        growth/tech y alternativos.
    """

    comunes = validar_datos(
        retornos,
        catalogo
    )

    retornos = (
        retornos[
            comunes
        ]
        .copy()
    )

    catalogo = (
        catalogo
        .set_index("ticker")
        .loc[comunes]
        .reset_index()
    )

    tickers = comunes

    n_activos = len(
        tickers
    )

    n_escenarios = len(
        retornos
    )

    mu = (
        retornos
        .mean()
        .to_numpy()
    )

    escenarios = (
        retornos
        .to_numpy()
    )


    # -----------------------------------------------------
    # PARÁMETROS DEL CLIENTE
    # -----------------------------------------------------

    penalizacion = (
        penalizacion_desde_lambda(
            lambda_cliente
        )
    )

    limites = (
        limites_desde_perfil(
            perfil_cliente
        )
    )

    peso_maximo = min(
        peso_maximo,
        limites[
            "peso_maximo"
        ]
    )

    max_alternativos = min(
        max_alternativos,
        limites[
            "max_alternativos"
        ]
    )

    min_bonos = limites[
        "min_bonos"
    ]

    max_equity = limites[
        "max_equity"
    ]

    max_growth_tech = limites[
        "max_growth_tech"
    ]


    # -----------------------------------------------------
    # MODELO
    # -----------------------------------------------------

    modelo = gp.Model(
        "behavioral_portfolio"
    )

    modelo.Params.OutputFlag = 0


    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    w = modelo.addVars(
        n_activos,
        lb=0.0,
        ub=peso_maximo,
        vtype=GRB.CONTINUOUS,
        name="w"
    )

    z = modelo.addVars(
        n_activos,
        vtype=GRB.BINARY,
        name="z"
    )

    eta = modelo.addVar(
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="VaR"
    )

    u = modelo.addVars(
        n_escenarios,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="exceso"
    )


    # -----------------------------------------------------
    # CAPITAL TOTAL
    # -----------------------------------------------------

    modelo.addConstr(
        gp.quicksum(
            w[i]
            for i in range(
                n_activos
            )
        ) == 1,
        name="capital_total"
    )


    # -----------------------------------------------------
    # NÚMERO MÁXIMO DE ACTIVOS
    # -----------------------------------------------------

    modelo.addConstr(
        gp.quicksum(
            z[i]
            for i in range(
                n_activos
            )
        ) <= max_activos,
        name="max_activos"
    )


    # -----------------------------------------------------
    # PESO MÍNIMO Y MÁXIMO
    # -----------------------------------------------------

    for i in range(
        n_activos
    ):

        modelo.addConstr(
            w[i]
            <= peso_maximo
            * z[i],
            name=f"peso_max_{i}"
        )

        modelo.addConstr(
            w[i]
            >= peso_minimo
            * z[i],
            name=f"peso_min_{i}"
        )


    sector_activo = (
        catalogo[
            "sector"
        ]
        .tolist()
    )

    clase_activo = (
        catalogo[
            "clase"
        ]
        .tolist()
    )


    # -----------------------------------------------------
    # LÍMITE POR SECTOR
    # -----------------------------------------------------

    for sector in (
        catalogo[
            "sector"
        ].unique()
    ):

        indices = [
            i
            for i in range(
                n_activos
            )
            if sector_activo[i]
            == sector
        ]

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i in indices
            ) <= max_sector,
            name=f"sector_{sector}"
        )


    # -----------------------------------------------------
    # LÍMITE POR CLASE
    # -----------------------------------------------------

    for clase in (
        catalogo[
            "clase"
        ].unique()
    ):

        indices = [
            i
            for i in range(
                n_activos
            )
            if clase_activo[i]
            == clase
        ]

        clase_norm = _texto(
            clase
        )

        if (
            "commodit" in clase_norm
            or "real estate"
            in clase_norm
        ):

            limite = (
                max_alternativos
            )

        else:

            limite = (
                max_clase
            )

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i in indices
            ) <= limite,
            name=f"clase_{clase}"
        )


    # =====================================================
    # SUITABILITY
    # =====================================================

    indices_bonos = (
        _indices_bonos(
            tickers,
            clase_activo
        )
    )

    if min_bonos > 0:

        if not indices_bonos:

            raise ValueError(
                "El perfil requiere bonos, pero "
                "el catálogo no contiene activos "
                "de renta fija reconocibles."
            )

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i
                in indices_bonos
            ) >= min_bonos,
            name="min_bonos_perfil"
        )


    indices_equity = (
        _indices_equity(
            tickers,
            clase_activo,
            sector_activo
        )
    )

    if indices_equity:

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i
                in indices_equity
            ) <= max_equity,
            name="max_equity_perfil"
        )


    indices_growth = (
        _indices_growth_tech(
            tickers,
            clase_activo,
            sector_activo
        )
    )

    if indices_growth:

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i
                in indices_growth
            ) <= max_growth_tech,
            name="max_growth_tech_perfil"
        )


    # =====================================================
    # CVaR
    # =====================================================

    for s in range(
        n_escenarios
    ):

        rendimiento_s = (
            gp.quicksum(
                escenarios[s, i]
                * w[i]
                for i in range(
                    n_activos
                )
            )
        )

        perdida_s = (
            -rendimiento_s
        )

        modelo.addConstr(
            u[s]
            >= perdida_s
            - eta,
            name=f"cvar_{s}"
        )


    cvar = (
        eta
        +
        (
            1
            /
            (
                (1 - alpha_cvar)
                * n_escenarios
            )
        )
        * gp.quicksum(
            u[s]
            for s
            in range(
                n_escenarios
            )
        )
    )


    rendimiento_esperado = (
        gp.quicksum(
            mu[i]
            * w[i]
            for i in range(
                n_activos
            )
        )
    )


    # -----------------------------------------------------
    # OBJETIVO
    # -----------------------------------------------------

    modelo.setObjective(
        rendimiento_esperado
        - penalizacion
        * cvar,
        GRB.MAXIMIZE
    )


    modelo.optimize()


    # -----------------------------------------------------
    # VALIDACIÓN
    # -----------------------------------------------------

    if modelo.Status != GRB.OPTIMAL:

        if modelo.Status == GRB.INFEASIBLE:

            raise ValueError(
                "Las restricciones del perfil "
                "hicieron el modelo infactible. "
                "Revisa el catálogo o los límites."
            )

        raise ValueError(
            "Gurobi no encontró una solución óptima."
        )


    # -----------------------------------------------------
    # RESULTADO
    # -----------------------------------------------------

    pesos = pd.Series(
        [
            w[i].X
            for i in range(
                n_activos
            )
        ],
        index=tickers,
        name="Peso"
    )

    pesos[
        np.abs(
            pesos
        ) < 1e-8
    ] = 0.0

    pesos = (
        pesos[
            pesos > 0
        ]
        .sort_values(
            ascending=False
        )
    )


    rendimiento_mensual = sum(
        mu[i]
        * w[i].X
        for i in range(
            n_activos
        )
    )

    rendimiento_anual = (
        (1 + rendimiento_mensual) ** 12
        - 1
    )


    return ResultadoPortafolio(
        pesos=pesos,
        rendimiento_mensual=float(
            rendimiento_mensual
        ),
        rendimiento_anual=float(
            rendimiento_anual
        ),
        cvar_mensual=float(
            cvar.getValue()
        ),
        lambda_cliente=float(
            lambda_cliente
        ),
        penalizacion_riesgo=float(
            penalizacion
        ),
        numero_activos=len(
            pesos
        ),
        estado="OPTIMAL"
    )


# =========================================================
# PRUEBA LOCAL
# =========================================================

if __name__ == "__main__":

    from datos import (
        cargar_catalogo,
        preparar_datos
    )

    print(
        "\nCargando información..."
    )

    catalogo = (
        cargar_catalogo()
    )

    precios, retornos = (
        preparar_datos()
    )

    resultado = (
        optimizar_portafolio(
            retornos=retornos,
            catalogo=catalogo,
            lambda_cliente=2.5,
            perfil_cliente="Moderado"
        )
    )

    print(
        "\nPORTAFOLIO RECOMENDADO"
    )

    print(
        resultado.pesos
    )

    print(
        "\nRendimiento mensual esperado:",
        f"{resultado.rendimiento_mensual:.2%}"
    )

    print(
        "Rendimiento anual aproximado:",
        f"{resultado.rendimiento_anual:.2%}"
    )

    print(
        "CVaR mensual 95%:",
        f"{resultado.cvar_mensual:.2%}"
    )