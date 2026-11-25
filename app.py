import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cuestionario import (
    PREGUNTAS,
    calcular_perfil
)

from datos import (
    cargar_catalogo,
    preparar_datos
)

from motor import (
    optimizar_portafolio,
    limites_desde_perfil
)

from simulacion import (
    evaluar_politicas,
    resumir_resultados
)


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title=(
        "Behavioral Portfolio Advisor"
    ),
    page_icon="📈",
    layout="wide"
)


# =========================================================
# ESTILO
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #98a2b3;
        margin-bottom: 2rem;
        max-width: 850px;
    }

    .profile-card,
    .portfolio-card {
        border: 1px solid #667085;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FUNCIONES
# =========================================================

@st.cache_data(
    show_spinner=False
)
def cargar_datos_mercado():

    catalogo = (
        cargar_catalogo()
    )

    precios, retornos = (
        preparar_datos(
            periodo="7y"
        )
    )

    return (
        catalogo,
        precios,
        retornos
    )


def nivel_lambda(lam):

    if lam <= 2:
        return "Baja"

    elif lam <= 3:
        return "Moderada"

    return "Alta"


def limpiar_simulacion():

    claves = [
        "resultados_simulacion",
        "resumen_simulacion",
        "banda_optima",
        "tuning_banda",
        "trayectorias",
    ]

    for clave in claves:

        if clave in (
            st.session_state
        ):

            del st.session_state[
                clave
            ]


def obtener_fila(
    resumen,
    politica
):

    fila = resumen[
        resumen[
            "politica"
        ] == politica
    ]

    if fila.empty:
        return None

    return fila.iloc[0]


# =========================================================
# TÍTULO
# =========================================================

st.markdown(
    """
    <p class="main-title">
    Behavioral Portfolio Advisor
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Conoce tu perfil de inversión, recibe un portafolio
    personalizado y observa cómo tus decisiones pueden
    afectar tu patrimonio a largo plazo.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PASO 1
# =========================================================

st.header(
    "1. Conoce tu perfil como inversionista"
)

st.write(
    """
    Responde de acuerdo con lo que realmente harías.
    No existen respuestas correctas o incorrectas.
    """
)


with st.form(
    "cuestionario"
):

    respuestas = {}

    for numero, pregunta in enumerate(
        PREGUNTAS,
        start=1
    ):

        st.markdown(
            f"### {numero}. {pregunta['titulo']}"
        )

        st.write(
            pregunta[
                "texto"
            ]
        )

        respuestas[
            pregunta[
                "id"
            ]
        ] = st.radio(
            "Selecciona una opción:",
            pregunta[
                "opciones"
            ],
            key=pregunta[
                "id"
            ],
            index=None
        )

        st.write("")


    enviar = (
        st.form_submit_button(
            "Calcular mi perfil",
            type="primary",
            use_container_width=True
        )
    )


if enviar:

    faltantes = any(
        respuesta is None
        for respuesta
        in respuestas.values()
    )

    if faltantes:

        st.error(
            "Responde todas las preguntas."
        )

    else:

        perfil = (
            calcular_perfil(
                respuestas
            )
        )

        st.session_state[
            "perfil"
        ] = perfil

        st.session_state[
            "respuestas"
        ] = respuestas


        if "portafolio" in (
            st.session_state
        ):

            del st.session_state[
                "portafolio"
            ]


        limpiar_simulacion()


# =========================================================
# PASO 2
# =========================================================

if "perfil" in (
    st.session_state
):

    perfil = (
        st.session_state[
            "perfil"
        ]
    )

    st.divider()

    st.header(
        "2. Tu perfil"
    )


    st.markdown(
        f"""
        <div class="profile-card">

        <h2>Perfil {perfil["perfil"]}</h2>

        <p>
        Tu perfil combina tu reacción frente al riesgo,
        sensibilidad a pérdidas y comportamientos comunes
        al invertir.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Aversión a pérdidas",
        nivel_lambda(
            perfil[
                "lambda"
            ]
        )
    )

    c2.metric(
        "Panic selling",
        perfil[
            "panic_level"
        ]
    )

    c3.metric(
        "Performance chasing",
        perfil[
            "chasing_level"
        ]
    )


    with st.expander(
        "Ver parámetros estimados"
    ):

        p1, p2, p3 = (
            st.columns(3)
        )

        p1.metric(
            "λ · Loss aversion",
            f'{perfil["lambda"]:.2f}'
        )

        p2.metric(
            "α · Gains",
            f'{perfil["alpha"]:.2f}'
        )

        p3.metric(
            "β · Losses",
            f'{perfil["beta"]:.2f}'
        )


        st.write(
            """
            λ representa sensibilidad a pérdidas.

            α representa la curvatura frente a ganancias.

            β representa la curvatura frente a pérdidas.
            """
        )


# =========================================================
# PASO 3
# =========================================================

    st.divider()

    st.header(
        "3. Construye tu portafolio"
    )

    st.write(
        """
        Utilizaremos tu perfil junto con información
        histórica para construir un portafolio personalizado.
        """
    )


    col1, col2 = (
        st.columns(2)
    )


    capital = (
        col1.number_input(
            "Capital a invertir",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=50_000
        )
    )


    max_activos = (
        col2.slider(
            "Máximo número de activos",
            min_value=5,
            max_value=12,
            value=8
        )
    )


    limites = (
        limites_desde_perfil(
            perfil[
                "perfil"
            ]
        )
    )


    with st.expander(
        "Ver restricciones aplicadas a tu perfil"
    ):

        st.write(
            f"""
            Para un perfil **{perfil["perfil"]}**:

            - mínimo en bonos: **{limites["min_bonos"]:.0%}**
            - máximo en renta variable: **{limites["max_equity"]:.0%}**
            - máximo conjunto en growth/tecnología: **{limites["max_growth_tech"]:.0%}**
            - máximo en activos alternativos: **{limites["max_alternativos"]:.0%}**
            - máximo por activo: **{limites["peso_maximo"]:.0%}**

            Estas restricciones funcionan como reglas de suitability.
            """
        )


    construir = (
        st.button(
            "Construir mi portafolio",
            type="primary",
            use_container_width=True
        )
    )


    if construir:

        try:

            with st.spinner(
                "Construyendo portafolio..."
            ):

                (
                    catalogo,
                    precios,
                    retornos
                ) = (
                    cargar_datos_mercado()
                )


                resultado = optimizar_portafolio(
                        retornos=retornos,
                        catalogo=catalogo,
                        lambda_cliente=perfil[
                            "lambda"
                        ],
                        perfil_cliente=perfil[
                            "perfil"
                        ],
                        max_activos=max_activos
                    )
                


                st.session_state[
                    "portafolio"
                ] = resultado

                st.session_state[
                    "capital"
                ] = capital

                st.session_state[
                    "catalogo"
                ] = catalogo

                st.session_state[
                    "numero_meses"
                ] = len(
                    retornos
                )

                st.session_state[
                    "max_activos"
                ] = max_activos


                limpiar_simulacion()


        except Exception as exc:

            st.error(
                f"No fue posible construir el portafolio: {exc}"
            )


# =========================================================
# PORTAFOLIO
# =========================================================

if (
    "perfil"
    in st.session_state
    and
    "portafolio"
    in st.session_state
):

    perfil = (
        st.session_state[
            "perfil"
        ]
    )

    resultado = (
        st.session_state[
            "portafolio"
        ]
    )

    capital = (
        st.session_state[
            "capital"
        ]
    )

    catalogo = (
        st.session_state[
            "catalogo"
        ]
    )


    st.divider()

    st.header(
        "4. Tu portafolio recomendado"
    )


    st.markdown(
        f"""
        <div class="portfolio-card">

        <h2>
        Portafolio para un perfil {perfil["perfil"]}
        </h2>

        <p>
        La recomendación combina rendimiento esperado,
        CVaR y restricciones de diversificación
        acordes con tu perfil.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = (
        st.columns(4)
    )


    m1.metric(
        "Rendimiento esperado anual",
        f"{resultado.rendimiento_anual:.2%}"
    )

    m2.metric(
        "CVaR mensual 95%",
        f"{resultado.cvar_mensual:.2%}"
    )

    m3.metric(
        "Activos seleccionados",
        resultado.numero_activos
    )

    m4.metric(
        "Capital",
        f"${capital:,.0f}"
    )


    tabla = (
        resultado.pesos
        .rename(
            "Peso"
        )
        .reset_index()
        .rename(
            columns={
                "index":
                "ticker"
            }
        )
    )


    tabla = (
        tabla.merge(
            catalogo,
            on="ticker",
            how="left"
        )
    )


    tabla[
        "Inversión"
    ] = (
        tabla[
            "Peso"
        ]
        * capital
    )


    columnas_tabla = [
        "ticker",
        "nombre",
        "clase",
        "sector",
        "Peso",
        "Inversión"
    ]


    tabla = tabla[
        columnas_tabla
    ]


    st.subheader(
        "Distribución recomendada"
    )


    st.dataframe(
        tabla.style.format(
            {
                "Peso":
                "{:.2%}",

                "Inversión":
                "${:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    fig_pesos = px.bar(
        tabla.sort_values(
            "Peso"
        ),
        x="Peso",
        y="ticker",
        orientation="h",
        text_auto=".1%",
        title=(
            "Distribución del portafolio"
        )
    )


    fig_pesos.update_layout(
        xaxis_tickformat=".0%",
        xaxis_title=(
            "Peso del portafolio"
        ),
        yaxis_title="",
        height=430
    )


    st.plotly_chart(
        fig_pesos,
        use_container_width=True
    )


    por_clase = (
        tabla
        .groupby(
            "clase",
            as_index=False
        )[
            "Peso"
        ]
        .sum()
    )


    fig_clase = px.pie(
        por_clase,
        values="Peso",
        names="clase",
        hole=0.55,
        title=(
            "Distribución por clase de activo"
        )
    )


    st.plotly_chart(
        fig_clase,
        use_container_width=True
    )


# =========================================================
# SIMULACIÓN
# =========================================================

    st.divider()

    st.header(
        "5. Tu futuro financiero a 10 años"
    )

    st.write(
        """
        Simulamos 500 posibles escenarios durante 120 meses.

        Todas las estrategias enfrentan exactamente
        los mismos escenarios.
        """
    )


    simular = (
        st.button(
            "Simular próximos 10 años",
            type="primary",
            use_container_width=True
        )
    )


    if simular:

        try:

            with st.spinner(
                "Simulando 500 escenarios..."
            ):

                (
                    catalogo_sim,
                    precios,
                    retornos
                ) = (
                    cargar_datos_mercado()
                )


                (
                    resultados_sim,
                    banda,
                    tuning,
                    trayectorias
                ) = (
                    evaluar_politicas(
                        retornos=retornos,
                        pesos=resultado.pesos,
                        perfil=perfil,
                        capital_inicial=capital,
                        n_escenarios=500,
                        catalogo=catalogo_sim,
                        max_activos=st.session_state.get(
                            "max_activos",
                            8
                        )
                    )
                )


                resumen = (
                    resumir_resultados(
                        resultados_sim
                    )
                )


                st.session_state[
                    "resultados_simulacion"
                ] = resultados_sim

                st.session_state[
                    "resumen_simulacion"
                ] = resumen

                st.session_state[
                    "banda_optima"
                ] = banda

                st.session_state[
                    "tuning_banda"
                ] = tuning

                st.session_state[
                    "trayectorias"
                ] = trayectorias


        except Exception as exc:

            st.error(
                f"Error durante la simulación: {exc}"
            )


# =========================================================
# RESULTADOS
# =========================================================

    if (
        "resumen_simulacion"
        in st.session_state
    ):

        resumen = (
            st.session_state[
                "resumen_simulacion"
            ]
        )

        trayectorias = (
            st.session_state[
                "trayectorias"
            ]
        )


        st.success(
            "Simulación completada."
        )


        st.subheader(
            "Comparación de estrategias"
        )


        tabla_resumen = resumen[
            [
                "politica",
                "riqueza_media",
                "riqueza_mediana",
                "percentil_5",
                "percentil_95",
                "IC95_low",
                "IC95_high",
                "drawdown_medio",
                "costos_medios"
            ]
        ].copy()


        tabla_resumen.columns = [
            "Estrategia",
            "Riqueza promedio",
            "Riqueza mediana",
            "Percentil 5",
            "Percentil 95",
            "IC 95% inferior",
            "IC 95% superior",
            "Drawdown promedio",
            "Costos promedio"
        ]


        st.dataframe(
            tabla_resumen.style.format(
                {
                    "Riqueza promedio":
                    "${:,.0f}",

                    "Riqueza mediana":
                    "${:,.0f}",

                    "Percentil 5":
                    "${:,.0f}",

                    "Percentil 95":
                    "${:,.0f}",

                    "IC 95% inferior":
                    "${:,.0f}",

                    "IC 95% superior":
                    "${:,.0f}",

                    "Drawdown promedio":
                    "{:.1%}",

                    "Costos promedio":
                    "${:,.0f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # PROSPECT VALUE
        # =================================================

        st.subheader(
            "Métrica principal: valor prospect"
        )

        st.write(
            """
            El objetivo utiliza la función de Prospect Theory
            del cliente aplicada al rendimiento total a 10 años.

            Un valor mayor representa un resultado más atractivo
            para este perfil.
            """
        )


        tabla_prospect = resumen[
            [
                "politica",
                "prospect_value_medio",
                "prospect_value_sd",
                "prospect_IC95_low",
                "prospect_IC95_high",
                "dif_prospect_vs_base",
                "dif_prospect_IC95_low",
                "dif_prospect_IC95_high"
            ]
        ].copy()


        tabla_prospect.columns = [
            "Estrategia",
            "Valor prospect medio",
            "Desviación estándar",
            "IC 95% inferior",
            "IC 95% superior",
            "Diferencia vs Buy & Hold",
            "Dif. IC 95% inferior",
            "Dif. IC 95% superior"
        ]


        st.dataframe(
            tabla_prospect.style.format(
                {
                    "Valor prospect medio":
                    "{:.3f}",

                    "Desviación estándar":
                    "{:.3f}",

                    "IC 95% inferior":
                    "{:.3f}",

                    "IC 95% superior":
                    "{:.3f}",

                    "Diferencia vs Buy & Hold":
                    "{:+.3f}",

                    "Dif. IC 95% inferior":
                    "{:+.3f}",

                    "Dif. IC 95% superior":
                    "{:+.3f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # RECOMENDACIÓN
        # =================================================

        st.subheader(
            "Nuestra recomendación"
        )


        politicas_recomendables = [
            "Buy & Hold",
            "Band Rebalancing (PFA)",
            "Lookahead (DLA)",
            "Re-optimización anual (CFA)"
        ]


        candidatas = (
            resumen[
                resumen[
                    "politica"
                ].isin(
                    politicas_recomendables
                )
            ]
            .sort_values(
                "prospect_value_medio",
                ascending=False
            )
        )


        mejor = (
            candidatas.iloc[0]
        )

        recomendada = (
            mejor[
                "politica"
            ]
        )


        # Si la ventaja contra Buy & Hold
        # no es estadísticamente clara,
        # preferimos la estrategia simple.
        if (
            recomendada
            != "Buy & Hold"
        ):

            if (
                mejor[
                    "dif_prospect_IC95_low"
                ] <= 0
                <= mejor[
                    "dif_prospect_IC95_high"
                ]
            ):

                recomendada = (
                    "Buy & Hold"
                )


        fila_rec = obtener_fila(
            resumen,
            recomendada
        )


        st.success(
            f"""
            Te recomendamos: **{recomendada}**.

            La riqueza mediana simulada después de 10 años es
            **${fila_rec["riqueza_mediana"]:,.0f}**.

            En el 5% de los escenarios más adversos,
            la riqueza final estaría por debajo de aproximadamente
            **${fila_rec["percentil_5"]:,.0f}**.
            """
        )


        # -------------------------------------------------
        # CUÁNDO CAMBIARÍA
        # -------------------------------------------------

        estrategias_disciplinadas = (
            resumen[
                resumen[
                    "politica"
                ].isin(
                    politicas_recomendables
                )
            ]
        )


        fila_menor_drawdown = (
            estrategias_disciplinadas
            .sort_values(
                "drawdown_medio"
            )
            .iloc[0]
        )


        st.markdown(
            "#### ¿Cuándo cambiaría esta recomendación?"
        )


        st.write(
            f"""
            - Si priorizas reducir caídas sobre crecimiento,
              entre las estrategias disciplinadas la menor caída
              promedio fue **{fila_menor_drawdown["politica"]}**
              con **{fila_menor_drawdown["drawdown_medio"]:.1%}**.

            - Si cambia tu perfil de riesgo o tu aversión a pérdidas,
              cambia el portafolio óptimo.

            - Si las condiciones futuras del mercado son distintas
              de las observadas históricamente, también puede cambiar
              la estrategia recomendada.
            """
        )


        # =================================================
        # FAN CHART
        # =================================================

        st.subheader(
            "Cómo podría evolucionar tu patrimonio"
        )


        estrategia_chart = (
            st.selectbox(
                "Estrategia",
                list(
                    trayectorias.keys()
                ),
                index=list(
                    trayectorias.keys()
                ).index(
                    recomendada
                )
            )
        )


        caminos = (
            trayectorias[
                estrategia_chart
            ]
        )


        meses = np.arange(
            caminos.shape[1]
        )

        anos = (
            meses
            / 12
        )


        p5 = np.percentile(
            caminos,
            5,
            axis=0
        )

        p25 = np.percentile(
            caminos,
            25,
            axis=0
        )

        p50 = np.percentile(
            caminos,
            50,
            axis=0
        )

        p75 = np.percentile(
            caminos,
            75,
            axis=0
        )

        p95 = np.percentile(
            caminos,
            95,
            axis=0
        )


        fig_fan = go.Figure()


        fig_fan.add_trace(
            go.Scatter(
                x=anos,
                y=p95,
                mode="lines",
                line=dict(
                    width=0
                ),
                showlegend=False
            )
        )


        fig_fan.add_trace(
            go.Scatter(
                x=anos,
                y=p5,
                mode="lines",
                fill="tonexty",
                name="Percentiles 5 - 95",
                line=dict(
                    width=0
                )
            )
        )


        fig_fan.add_trace(
            go.Scatter(
                x=anos,
                y=p75,
                mode="lines",
                line=dict(
                    width=0
                ),
                showlegend=False
            )
        )


        fig_fan.add_trace(
            go.Scatter(
                x=anos,
                y=p25,
                mode="lines",
                fill="tonexty",
                name="Percentiles 25 - 75",
                line=dict(
                    width=0
                )
            )
        )


        fig_fan.add_trace(
            go.Scatter(
                x=anos,
                y=p50,
                mode="lines",
                name="Mediana",
                line=dict(
                    width=3
                )
            )
        )


        for i in range(
            min(
                8,
                len(
                    caminos
                )
            )
        ):

            fig_fan.add_trace(
                go.Scatter(
                    x=anos,
                    y=caminos[i],
                    mode="lines",
                    line=dict(
                        width=1
                    ),
                    opacity=0.20,
                    showlegend=(
                        i == 0
                    ),
                    name=(
                        "Escenarios de ejemplo"
                        if i == 0
                        else None
                    )
                )
            )


        fig_fan.update_layout(
            title=(
                f"Riqueza simulada — {estrategia_chart}"
            ),
            xaxis_title="Años",
            yaxis_title="Riqueza",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            height=520
        )


        st.plotly_chart(
            fig_fan,
            use_container_width=True
        )


        # =================================================
        # RIQUEZA MEDIANA
        # =================================================

        fig_mediana = px.bar(
            tabla_resumen.sort_values(
                "Riqueza mediana"
            ),
            x="Riqueza mediana",
            y="Estrategia",
            orientation="h",
            text_auto="$,.0f",
            title=(
                "Riqueza mediana después de 10 años"
            )
        )


        fig_mediana.update_layout(
            xaxis_tickprefix="$",
            xaxis_tickformat=",.0f",
            yaxis_title="",
            height=430
        )


        st.plotly_chart(
            fig_mediana,
            use_container_width=True
        )


        # =================================================
        # RIESGO VS RIQUEZA
        # =================================================

        st.subheader(
            "Rendimiento y riesgo"
        )


        fig_riesgo = px.scatter(
            tabla_resumen,
            x="Drawdown promedio",
            y="Riqueza mediana",
            text="Estrategia",
            size="Riqueza promedio",
            title=(
                "Riqueza esperada vs. caídas del portafolio"
            )
        )


        fig_riesgo.update_traces(
            textposition="top center"
        )


        fig_riesgo.update_layout(
            xaxis_tickformat=".1%",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            height=500
        )


        st.plotly_chart(
            fig_riesgo,
            use_container_width=True
        )


        st.caption(
            f"""
            Band Rebalancing seleccionó una banda de
            {st.session_state["banda_optima"]:.0%}
            utilizando escenarios separados de tuning.
            """
        )


        # =================================================
        # DIAGNÓSTICO CFA
        # =================================================

        fila_cfa = obtener_fila(
            resumen,
            "Re-optimización anual (CFA)"
        )


        if (
            fila_cfa is not None
            and fila_cfa[
                "cfa_fallos_total"
            ] > 0
        ):

            st.warning(
                f"""
                CFA registró
                {int(fila_cfa["cfa_fallos_total"])}
                fallos de re-optimización en
                {int(fila_cfa["cfa_escenarios_con_fallo"])}
                escenarios.

                En esos casos se utilizó temporalmente
                el portafolio base.
                """
            )


        # =================================================
        # BEHAVIORAL IMPACT
        # =================================================

        st.divider()

        st.header(
            "6. El impacto de tu comportamiento"
        )


        fila_ref = obtener_fila(
            resumen,
            recomendada
        )

        fila_panic = obtener_fila(
            resumen,
            "Panic Selling"
        )

        fila_chasing = obtener_fila(
            resumen,
            "Performance Chasing"
        )


        referencia = float(
            fila_ref[
                "riqueza_mediana"
            ]
        )


        panic = float(
            fila_panic[
                "riqueza_mediana"
            ]
        )


        chasing = float(
            fila_chasing[
                "riqueza_mediana"
            ]
        )


        impacto_panic = (
            panic
            - referencia
        )


        impacto_chasing = (
            chasing
            - referencia
        )


        b1, b2 = (
            st.columns(2)
        )


        b1.metric(
            "Impacto de Panic Selling",
            f"${impacto_panic:,.0f}",
            delta=(
                f"{impacto_panic / referencia:.1%}"
            )
        )


        b2.metric(
            "Impacto de Performance Chasing",
            f"${impacto_chasing:,.0f}",
            delta=(
                f"{impacto_chasing / referencia:.1%}"
            )
        )


        st.write(
            f"""
            Estas cifras comparan los comportamientos
            contra la estrategia recomendada para tu perfil:
            **{recomendada}**.

            Todas las estrategias enfrentan exactamente
            los mismos escenarios.
            """
        )


        if impacto_panic < 0:

            st.warning(
                f"""
                Reaccionar vendiendo después de caídas
                redujo la riqueza mediana en aproximadamente
                **${abs(impacto_panic):,.0f}**
                frente a **{recomendada}**.
                """
            )


        if impacto_chasing < 0:

            st.warning(
                f"""
                Perseguir activos con buen desempeño reciente
                redujo la riqueza mediana en aproximadamente
                **${abs(impacto_chasing):,.0f}**
                frente a **{recomendada}**.
                """
            )


        # =================================================
        # METODOLOGÍA
        # =================================================

        with st.expander(
            "Ver metodología de la simulación"
        ):

            st.markdown(
                f"""
                ### Powell model

                **State**

                - riqueza actual
                - pesos actuales
                - máximo histórico de riqueza
                - rendimientos recientes

                **Decision**

                Los trades o cambios de pesos realizados
                por cada política.

                **Exogenous information**

                Rendimientos mensuales del mercado.

                **Transition**

                La riqueza cambia con los retornos realizados
                menos costos de transacción.

                **Objective**

                Maximizar el valor esperado de Prospect Theory
                del patrimonio final respecto al capital inicial.

                ### Portfolio construction

                El MILP Mean-CVaR utiliza:

                - λ para controlar la penalización por CVaR;
                - el perfil {perfil["perfil"]} para controlar
                  restricciones de suitability;
                - máximo por activo;
                - límites por sector y clase;
                - mínimo de renta fija;
                - límite agregado a growth/technology.

                ### Simulation

                - 500 escenarios
                - 120 meses
                - block bootstrap de 3 a 6 meses
                - common random numbers
                - seeds separados para tuning y evaluación
                - 10 bps de costo sobre valor negociado

                ### Policies

                - Buy & Hold
                - Band Rebalancing (PFA)
                - Lookahead (DLA)
                - Re-optimización anual (CFA)
                - Panic Selling
                - Performance Chasing
                """
            )
