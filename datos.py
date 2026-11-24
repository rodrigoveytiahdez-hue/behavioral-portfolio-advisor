from pathlib import Path
import json

import pandas as pd
import yfinance as yf


BASE = Path(__file__).resolve().parent
CATALOGO = BASE / "activos.json"
DATA_DIR = BASE / "data"


def cargar_catalogo():
    """
    Lee el catálogo de ETFs definido en activos.json.
    """
    with open(CATALOGO, "r", encoding="utf-8") as f:
        contenido = json.load(f)

    return pd.DataFrame(contenido["activos"])


def obtener_tickers():
    catalogo = cargar_catalogo()

    return catalogo["ticker"].tolist()


def descargar_precios(
    tickers=None,
    periodo="7y"
):
    """
    Descarga precios ajustados desde Yahoo Finance.

    Se utilizan 7 años para asegurar que tengamos
    al menos 5 años completos después de limpieza.
    """

    if tickers is None:
        tickers = obtener_tickers()

    datos = yf.download(
        tickers,
        period=periodo,
        auto_adjust=True,
        progress=False
    )

    if datos.empty:
        raise ValueError(
            "No se pudieron descargar datos desde Yahoo Finance."
        )

    # Cuando hay múltiples activos, yfinance devuelve
    # columnas multinivel.
    if isinstance(datos.columns, pd.MultiIndex):
        precios = datos["Close"].copy()
    else:
        precios = datos[["Close"]].copy()
        precios.columns = tickers

    precios = precios.sort_index()

    # Eliminamos activos completamente vacíos
    precios = precios.dropna(
        axis=1,
        how="all"
    )

    return precios


def convertir_a_mensual(precios):
    """
    Convierte precios diarios a precios de cierre mensual.
    """

    precios_mensuales = precios.resample(
        "ME"
    ).last()

    return precios_mensuales


def calcular_retornos_mensuales(precios):
    """
    Calcula rendimientos mensuales:

    r_t = P_t / P_(t-1) - 1
    """

    precios_mensuales = convertir_a_mensual(
        precios
    )

    # Si el último mes todavía no termina (por ejemplo, hoy es
    # 23 de noviembre), su "cierre mensual" es solo un cierre
    # parcial. Lo quitamos para no mezclar meses incompletos.
    ultimo_dia = precios.index[-1]

    if ultimo_dia < precios_mensuales.index[-1] - pd.Timedelta(days=3):
        precios_mensuales = precios_mensuales.iloc[:-1]

    retornos = precios_mensuales.pct_change(
        fill_method=None
    )

    retornos = retornos.dropna(
        how="any"
    )

    return retornos


def preparar_datos(
    tickers=None,
    periodo="7y"
):
    """
    Ejecuta el flujo completo de preparación.
    """

    precios = descargar_precios(
        tickers=tickers,
        periodo=periodo
    )

    retornos = calcular_retornos_mensuales(
        precios
    )

    return precios, retornos


def cargar_datos_guardados():
    """
    Lee los CSV de la carpeta data/.

    Se usa como respaldo cuando Yahoo Finance no responde,
    para que el dashboard funcione aunque no haya internet
    (por ejemplo, durante la presentación).
    """

    precios = pd.read_csv(
        DATA_DIR / "precios_diarios.csv",
        index_col=0,
        parse_dates=True
    )

    # Recalculamos los rendimientos con la misma función
    # para aplicar la misma limpieza (quitar meses incompletos).
    retornos = calcular_retornos_mensuales(
        precios
    )

    return precios, retornos


def guardar_datos(
    precios,
    retornos
):
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    precios.to_csv(
        DATA_DIR / "precios_diarios.csv"
    )

    retornos.to_csv(
        DATA_DIR / "retornos_mensuales.csv"
    )


if __name__ == "__main__":

    catalogo = cargar_catalogo()

    print("\nActivos del proyecto:")
    print(
        catalogo[
            ["ticker", "clase", "sector"]
        ]
    )

    print("\nDescargando precios...")

    precios, retornos = preparar_datos()

    guardar_datos(
        precios,
        retornos
    )

    print("\nPrecios:")
    print(precios.tail())

    print("\nRendimientos mensuales:")
    print(retornos.tail())

    print(
        "\nNúmero de meses disponibles:",
        len(retornos)
    )

    print(
        "\nNúmero de activos:",
        retornos.shape[1]
    )