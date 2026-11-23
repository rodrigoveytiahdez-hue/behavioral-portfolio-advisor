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


def penalizacion_desde_lambda(lam):
    """
    Convierte la aversión a pérdidas del cliente en el peso
    que damos al CVaR dentro de la función objetivo.

    Lambda bajo  -> cliente tolera más pérdidas.
    Lambda alto  -> cliente es más sensible a pérdidas.
    """

    lam = float(lam)

    # Nuestro cuestionario produce aproximadamente valores
    # entre 1.5 y 4.5.
    lam = np.clip(lam, 1.5, 4.5)

    minimo = 0.20
    maximo = 1.20

    normalizado = (lam - 1.5) / (4.5 - 1.5)

    return minimo + normalizado * (maximo - minimo)


def validar_datos(retornos, catalogo):
    """
    Comprueba que los datos necesarios para optimizar existen.
    """

    if retornos.empty:
        raise ValueError(
            "No existen rendimientos para construir el portafolio."
        )

    if len(retornos) < 60:
        raise ValueError(
            "Se necesitan al menos 60 meses de información."
        )

    columnas_catalogo = set(catalogo["ticker"])
    columnas_retornos = set(retornos.columns)

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


def optimizar_portafolio(
    retornos,
    catalogo,
    lambda_cliente,
    max_activos=8,
    peso_minimo=0.05,
    peso_maximo=0.30,
    max_sector=0.40,
    alpha_cvar=0.95,
):
    """
    Construye un portafolio mediante un modelo Mean-CVaR.

    Variables:
        w_i = peso del activo i
        z_i = 1 si el activo forma parte del portafolio

    Restricciones:
        - suma de pesos = 1
        - no short selling
        - máximo K activos
        - peso mínimo y máximo por activo
        - exposición máxima por sector

    Objetivo:
        maximizar rendimiento esperado
        menos una penalización por CVaR.
    """

    comunes = validar_datos(
        retornos,
        catalogo
    )

    retornos = retornos[comunes].copy()

    catalogo = (
        catalogo
        .set_index("ticker")
        .loc[comunes]
        .reset_index()
    )

    tickers = comunes

    n_activos = len(tickers)
    n_escenarios = len(retornos)

    # Rendimiento medio mensual de cada activo
    mu = retornos.mean().to_numpy()

    # Matriz:
    # filas = meses históricos
    # columnas = activos
    escenarios = retornos.to_numpy()

    penalizacion = penalizacion_desde_lambda(
        lambda_cliente
    )

    modelo = gp.Model(
        "behavioral_portfolio"
    )

    # Para que Gurobi no llene la terminal de mensajes.
    modelo.Params.OutputFlag = 0


    # --------------------------------------------------
    # VARIABLES DE DECISIÓN
    # --------------------------------------------------

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


    # VaR
    eta = modelo.addVar(
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="VaR"
    )


    # Exceso de pérdida sobre VaR para cada escenario
    u = modelo.addVars(
        n_escenarios,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="exceso"
    )


    # --------------------------------------------------
    # RESTRICCIONES DEL PORTAFOLIO
    # --------------------------------------------------

    modelo.addConstr(
        gp.quicksum(
            w[i]
            for i in range(n_activos)
        ) == 1,
        name="capital_total"
    )


    # Máximo número de activos
    modelo.addConstr(
        gp.quicksum(
            z[i]
            for i in range(n_activos)
        ) <= max_activos,
        name="max_activos"
    )


    # Relación entre selección y peso
    for i in range(n_activos):

        modelo.addConstr(
            w[i] <= peso_maximo * z[i],
            name=f"peso_max_{i}"
        )

        modelo.addConstr(
            w[i] >= peso_minimo * z[i],
            name=f"peso_min_{i}"
        )


    # --------------------------------------------------
    # LÍMITE POR SECTOR
    # --------------------------------------------------

    sectores = catalogo["sector"].unique()

    for sector in sectores:

        indices = [
            i
            for i in range(n_activos)
            if catalogo.iloc[i]["sector"] == sector
        ]

        modelo.addConstr(
            gp.quicksum(
                w[i]
                for i in indices
            ) <= max_sector,
            name=f"sector_{sector}"
        )


    # --------------------------------------------------
    # CVaR
    # --------------------------------------------------

    for s in range(n_escenarios):

        # Rendimiento del portafolio en el escenario s
        rendimiento_s = gp.quicksum(
            escenarios[s, i] * w[i]
            for i in range(n_activos)
        )

        # Pérdida = -rendimiento
        perdida_s = -rendimiento_s

        modelo.addConstr(
            u[s] >= perdida_s - eta,
            name=f"cvar_{s}"
        )


    cvar = (
        eta
        +
        (1 / ((1 - alpha_cvar) * n_escenarios))
        * gp.quicksum(
            u[s]
            for s in range(n_escenarios)
        )
    )


    # --------------------------------------------------
    # RENDIMIENTO ESPERADO
    # --------------------------------------------------

    rendimiento_esperado = gp.quicksum(
        mu[i] * w[i]
        for i in range(n_activos)
    )


    # --------------------------------------------------
    # FUNCIÓN OBJETIVO
    # --------------------------------------------------

    modelo.setObjective(
        rendimiento_esperado
        - penalizacion * cvar,
        GRB.MAXIMIZE
    )


    modelo.optimize()


    # --------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------

    if modelo.Status != GRB.OPTIMAL:
        raise ValueError(
            "Gurobi no encontró una solución óptima."
        )


    pesos = pd.Series(
        [
            w[i].X
            for i in range(n_activos)
        ],
        index=tickers,
        name="Peso"
    )


    # Eliminamos errores numéricos diminutos
    pesos[
        np.abs(pesos) < 1e-8
    ] = 0.0


    pesos = pesos[
        pesos > 0
    ].sort_values(
        ascending=False
    )


    rendimiento_mensual = sum(
        mu[i] * w[i].X
        for i in range(n_activos)
    )


    rendimiento_anual = (
        (1 + rendimiento_mensual) ** 12
        - 1
    )


    resultado = ResultadoPortafolio(
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
        numero_activos=len(pesos),
        estado="OPTIMAL"
    )

    return resultado

if __name__ == "__main__":

    from datos import (
        cargar_catalogo,
        preparar_datos
    )

    print(
        "\nCargando información..."
    )

    catalogo = cargar_catalogo()

    precios, retornos = preparar_datos()


    # Cliente de prueba:
    # lambda = 2.5
    resultado = optimizar_portafolio(
        retornos=retornos,
        catalogo=catalogo,
        lambda_cliente=2.5
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


    print(
        "Penalización por riesgo:",
        f"{resultado.penalizacion_riesgo:.2f}"
    )