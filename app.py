import streamlit as st
import pandas as pd
import plotly.express as px

from cuestionario import PREGUNTAS, calcular_perfil
from datos import cargar_catalogo, preparar_datos
from motor import optimizar_portafolio
from simulacion import evaluar_politicas, resumir_resultados


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Behavioral Portfolio Advisor",
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
        color: #667085;
        margin-bottom: 2rem;
        max-width: 850px;
    }

    .profile-card {
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .portfolio-card {
        border: 1px solid #d0d5dd;
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
# FUNCIONES AUXILIARES
# =========================================================

@st.cache_data(show_spinner=False)
def cargar_datos_mercado():

    catalogo = cargar_catalogo()

    precios, retornos = preparar_datos(
        periodo="7y"
    )

    return catalogo, precios, retornos


def nivel_lambda(lam):

    if lam <= 2.0:
        return "Baja"

    elif lam <= 3.0:
        return "Moderada"

    return "Alta"


def limpiar_simulacion():

    claves = [
        "resultados_simulacion",
        "resumen_simulacion",
        "banda_optima",
        "tuning_banda"
    ]

    for clave in claves:

        if clave in st.session_state:
            del st.session_state[clave]


# =========================================================
# ENCABEZADO
# =========================================================

st.markdown(
    '<p class="main-title">Behavioral Portfolio Advisor</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Conoce tu perfil de inversión, recibe un portafolio personalizado
    y descubre cómo tus propias decisiones pueden afectar tu patrimonio
    a largo plazo.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PASO 1 — CUESTIONARIO
# =========================================================

st.header("1. Conoce tu perfil como inversionista")

st.write(
    """
    Responde de acuerdo con lo que realmente harías.
    No existen respuestas correctas o incorrectas.
    """
)


with st.form("cuestionario"):

    respuestas = {}

    for numero, pregunta in enumerate(
        PREGUNTAS,
        start=1
    ):

        st.markdown(
            f"### {numero}. {pregunta['titulo']}"
        )

        st.write(
            pregunta["texto"]
        )

        respuestas[pregunta["id"]] = st.radio(
            "Selecciona una opción:",
            pregunta["opciones"],
            key=pregunta["id"],
            index=None
        )

        st.write("")

    enviar = st.form_submit_button(
        "Calcular mi perfil",
        type="primary",
        use_container_width=True
    )


# =========================================================
# CALCULAR PERFIL
# =========================================================

if enviar:

    faltantes = any(
        respuesta is None
        for respuesta in respuestas.values()
    )

    if faltantes:

        st.error(
            "Por favor responde todas las preguntas antes de continuar."
        )

    else:

        perfil = calcular_perfil(
            respuestas
        )

        st.session_state["perfil"] = perfil
        st.session_state["respuestas"] = respuestas

        # Si cambia el perfil, eliminamos resultados anteriores
        if "portafolio" in st.session_state:
            del st.session_state["portafolio"]

        limpiar_simulacion()


# =========================================================
# PASO 2 — PERFIL
# =========================================================

if "perfil" in st.session_state:

    perfil = st.session_state["perfil"]

    st.divider()

    st.header("2. Tu perfil")

    st.markdown(
        f"""
        <div class="profile-card">

            <h2>Perfil {perfil["perfil"]}</h2>

            <p>
            Tu perfil combina tu reacción frente al riesgo,
            tu sensibilidad a las pérdidas y algunos
            comportamientos comunes al invertir.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # MÉTRICAS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Aversión a pérdidas",
        nivel_lambda(
            perfil["lambda"]
        )
    )

    c2.metric(
        "Panic selling",
        perfil["panic_level"]
    )

    c3.metric(
        "Performance chasing",
        perfil["chasing_level"]
    )


    # -----------------------------------------------------
    # PARÁMETROS
    # -----------------------------------------------------

    with st.expander(
        "Ver parámetros estimados"
    ):

        p1, p2, p3 = st.columns(3)

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
            **λ** representa qué tan sensibles somos a las pérdidas
            frente a ganancias de tamaño similar.

            **α** representa cómo evaluamos ganancias inciertas.

            **β** representa cómo evaluamos pérdidas inciertas.

            Estos valores son aproximaciones obtenidas mediante
            el cuestionario.
            """
        )


    # =====================================================
    # PASO 3 — CONSTRUIR PORTAFOLIO
    # =====================================================

    st.divider()

    st.header("3. Construye tu portafolio")

    st.write(
        """
        Utilizaremos tu perfil junto con información histórica
        de diferentes clases de activos para construir un
        portafolio personalizado.
        """
    )


    col_capital, col_activos = st.columns(2)

    capital = col_capital.number_input(
        "Capital a invertir",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=50_000,
        format="%d"
    )

    max_activos = col_activos.slider(
        "Máximo número de activos",
        min_value=5,
        max_value=12,
        value=8
    )


    st.caption(
        """
        Para esta demostración utilizamos más de cinco años
        de información histórica de ETFs de acciones,
        bonos, mercados internacionales, bienes raíces
        y commodities.
        """
    )


    construir = st.button(
        "Construir mi portafolio",
        type="primary",
        use_container_width=True
    )


    if construir:

        try:

            with st.spinner(
                "Analizando el mercado y construyendo tu portafolio..."
            ):

                catalogo, precios, retornos = (
                    cargar_datos_mercado()
                )

                resultado = optimizar_portafolio(
                    retornos=retornos,
                    catalogo=catalogo,
                    lambda_cliente=perfil["lambda"],
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
                ] = len(retornos)

                limpiar_simulacion()

        except Exception as exc:

            st.error(
                f"No fue posible construir el portafolio: {exc}"
            )


# =========================================================
# PASO 4 — PORTAFOLIO
# =========================================================

if (
    "perfil" in st.session_state
    and
    "portafolio" in st.session_state
):

    resultado = st.session_state[
        "portafolio"
    ]

    perfil = st.session_state[
        "perfil"
    ]

    capital = st.session_state[
        "capital"
    ]

    catalogo = st.session_state[
        "catalogo"
    ]


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
        La recomendación busca balancear crecimiento y protección
        frente a escenarios de pérdidas fuertes, tomando en cuenta
        tu nivel de aversión a las pérdidas.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

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


    # -----------------------------------------------------
    # TABLA
    # -----------------------------------------------------

    tabla = (
        resultado.pesos
        .rename("Peso")
        .reset_index()
        .rename(
            columns={
                "index": "ticker"
            }
        )
    )


    tabla = tabla.merge(
        catalogo,
        on="ticker",
        how="left"
    )


    tabla["Inversión"] = (
        tabla["Peso"]
        * capital
    )


    tabla = tabla[
        [
            "ticker",
            "nombre",
            "clase",
            "sector",
            "Peso",
            "Inversión"
        ]
    ]


    st.subheader(
        "Distribución recomendada"
    )


    st.dataframe(
        tabla.style.format(
            {
                "Peso": "{:.2%}",
                "Inversión": "${:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # GRÁFICA DE PESOS
    # -----------------------------------------------------

    fig = px.bar(
        tabla.sort_values(
            "Peso",
            ascending=True
        ),
        x="Peso",
        y="ticker",
        orientation="h",
        text_auto=".1%",
        title="Distribución del portafolio",
        hover_data=[
            "nombre",
            "clase",
            "sector"
        ]
    )


    fig.update_layout(
        xaxis_tickformat=".0%",
        xaxis_title="Peso del portafolio",
        yaxis_title="",
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DISTRIBUCIÓN POR CLASE
    # -----------------------------------------------------

    por_clase = (
        tabla
        .groupby(
            "clase",
            as_index=False
        )["Peso"]
        .sum()
    )


    fig_clase = px.pie(
        por_clase,
        values="Peso",
        names="clase",
        hole=0.55,
        title="Distribución por clase de activo"
    )


    st.plotly_chart(
        fig_clase,
        use_container_width=True
    )


    # -----------------------------------------------------
    # EXPLICACIÓN
    # -----------------------------------------------------

    st.subheader(
        "¿Por qué este portafolio?"
    )


    if perfil["lambda"] >= 3.5:

        st.info(
            """
            Tus respuestas muestran una sensibilidad relativamente
            alta a las pérdidas. Por esta razón, el modelo da mayor
            importancia a limitar pérdidas en escenarios adversos.
            """
        )

    elif perfil["lambda"] >= 2.5:

        st.info(
            """
            Tus respuestas muestran una sensibilidad moderada a las
            pérdidas. El portafolio busca un balance entre crecimiento
            y protección frente a escenarios negativos.
            """
        )

    else:

        st.info(
            """
            Tus respuestas muestran una mayor tolerancia a las pérdidas.
            El modelo puede aceptar más riesgo en busca de un mayor
            potencial de crecimiento.
            """
        )


    st.caption(
        f"""
        La estimación utiliza aproximadamente
        {st.session_state["numero_meses"]} meses de información histórica.
        Los rendimientos históricos no garantizan resultados futuros.
        """
    )


    # =====================================================
    # PASO 5 — SIMULACIÓN A 10 AÑOS
    # =====================================================

    st.divider()

    st.header(
        "5. Tu futuro financiero a 10 años"
    )

    st.write(
        """
        Simularemos 500 posibles escenarios de mercado durante
        los próximos 10 años.

        Todas las estrategias enfrentarán exactamente los mismos
        escenarios para poder compararlas de forma justa.
        """
    )


    simular = st.button(
        "Simular próximos 10 años",
        type="primary",
        use_container_width=True
    )


    if simular:

        try:

            with st.spinner(
                "Simulando 500 escenarios y comparando estrategias..."
            ):

                catalogo_sim, precios, retornos = (
                    cargar_datos_mercado()
                )

                (
                    resultados_sim,
                    banda,
                    tuning
                ) = evaluar_politicas(
                    retornos=retornos,
                    pesos=resultado.pesos,
                    perfil=perfil,
                    capital_inicial=capital,
                    n_escenarios=500
                )

                resumen_sim = (
                    resumir_resultados(
                        resultados_sim
                    )
                )

                st.session_state[
                    "resultados_simulacion"
                ] = resultados_sim

                st.session_state[
                    "resumen_simulacion"
                ] = resumen_sim

                st.session_state[
                    "banda_optima"
                ] = banda

                st.session_state[
                    "tuning_banda"
                ] = tuning


        except Exception as exc:

            st.error(
                f"No fue posible ejecutar la simulación: {exc}"
            )


    # =====================================================
    # RESULTADOS DE LA SIMULACIÓN
    # =====================================================

    if (
        "resultados_simulacion"
        in st.session_state
        and
        "resumen_simulacion"
        in st.session_state
    ):

        resultados_sim = st.session_state[
            "resultados_simulacion"
        ]

        resumen_sim = st.session_state[
            "resumen_simulacion"
        ]


        st.success(
            "Simulación completada."
        )


        # -------------------------------------------------
        # RESUMEN
        # -------------------------------------------------

        st.subheader(
            "Comparación de estrategias"
        )


        tabla_resumen = resumen_sim[
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
                    "Riqueza promedio": "${:,.0f}",
                    "Riqueza mediana": "${:,.0f}",
                    "Percentil 5": "${:,.0f}",
                    "Percentil 95": "${:,.0f}",
                    "IC 95% inferior": "${:,.0f}",
                    "IC 95% superior": "${:,.0f}",
                    "Drawdown promedio": "{:.1%}",
                    "Costos promedio": "${:,.0f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # RIQUEZA MEDIANA
        # -------------------------------------------------

        grafica = (
            tabla_resumen
            .sort_values(
                "Riqueza mediana"
            )
        )


        fig_riqueza = px.bar(
            grafica,
            x="Riqueza mediana",
            y="Estrategia",
            orientation="h",
            text_auto="$,.0f",
            title=(
                "Riqueza mediana después de 10 años"
            )
        )


        fig_riqueza.update_layout(
            xaxis_title="Riqueza final",
            yaxis_title="",
            height=450
        )


        st.plotly_chart(
            fig_riqueza,
            use_container_width=True
        )


        # -------------------------------------------------
        # RIESGO VS RIQUEZA
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PFA TUNING
        # -------------------------------------------------

        st.caption(
            f"""
            La política Band Rebalancing seleccionó una
            banda óptima de
            {st.session_state["banda_optima"]:.0%}
            utilizando escenarios separados de tuning.
            """
        )


        # =================================================
        # PASO 6 — BEHAVIORAL IMPACT
        # =================================================

        st.divider()

        st.header(
            "6. El impacto de tu comportamiento"
        )


        def mediana_policy(nombre):

            fila = resumen_sim[
                resumen_sim[
                    "politica"
                ] == nombre
            ]

            if fila.empty:
                return None

            return float(
                fila.iloc[0][
                    "riqueza_mediana"
                ]
            )


        disciplinado = mediana_policy(
            "Band Rebalancing (PFA)"
        )

        panic = mediana_policy(
            "Panic Selling"
        )

        chasing = mediana_policy(
            "Performance Chasing"
        )


        impacto_panic = None

        if (
            panic is not None
            and disciplinado is not None
        ):

            impacto_panic = (
                panic
                - disciplinado
            )


        impacto_chasing = None

        if (
            chasing is not None
            and disciplinado is not None
        ):

            impacto_chasing = (
                chasing
                - disciplinado
            )


        b1, b2 = st.columns(2)


        if impacto_panic is not None:

            b1.metric(
                "Impacto de Panic Selling",
                f"${impacto_panic:,.0f}",
                delta=(
                    f"{impacto_panic / disciplinado:.1%}"
                )
            )


        if impacto_chasing is not None:

            b2.metric(
                "Impacto de Performance Chasing",
                f"${impacto_chasing:,.0f}",
                delta=(
                    f"{impacto_chasing / disciplinado:.1%}"
                )
            )


        st.write(
            """
            Estas cifras comparan las estrategias de comportamiento
            con una estrategia de rebalanceo disciplinado utilizando
            exactamente los mismos escenarios de mercado.
            """
        )


        # -------------------------------------------------
        # PANIC SELLING
        # -------------------------------------------------

        if impacto_panic is not None:

            if impacto_panic < 0:

                st.warning(
                    f"""
                    En nuestra simulación, reaccionar vendiendo
                    después de caídas importantes redujo la riqueza
                    mediana después de 10 años en aproximadamente
                    **${abs(impacto_panic):,.0f}**.
                    """
                )

            else:

                st.info(
                    f"""
                    En estos escenarios históricos simulados,
                    Panic Selling terminó con aproximadamente
                    **${impacto_panic:,.0f} más** que el
                    rebalanceo disciplinado.

                    Esto no significa que vender durante caídas
                    sea necesariamente una estrategia superior.
                    El resultado depende de cómo evolucione
                    posteriormente el mercado.
                    """
                )


        # -------------------------------------------------
        # PERFORMANCE CHASING
        # -------------------------------------------------

        if impacto_chasing is not None:

            if impacto_chasing < 0:

                st.warning(
                    f"""
                    Perseguir los activos que recientemente tuvieron
                    mejor desempeño redujo la riqueza mediana en
                    aproximadamente
                    **${abs(impacto_chasing):,.0f}**.
                    """
                )

            else:

                st.info(
                    f"""
                    En estos escenarios, Performance Chasing terminó
                    con aproximadamente
                    **${impacto_chasing:,.0f} más**
                    de riqueza mediana que el rebalanceo disciplinado.

                    Sin embargo, esta comparación también debe
                    considerar el nivel de riesgo y drawdown.
                    """
                )


        # =================================================
        # METODOLOGÍA PARA EL PROFESOR
        # =================================================

        with st.expander(
            "Ver metodología de la simulación"
        ):

            st.markdown(
                """
                ### Modelo secuencial

                Cada mes el estado contiene:

                - riqueza actual;
                - pesos actuales;
                - máximo histórico de riqueza;
                - retornos recientes.

                La decisión corresponde a los trades que realiza
                cada política.

                La información exógena son los rendimientos
                mensuales del mercado.

                ### Escenarios

                Se utilizan 500 escenarios de 120 meses.

                Los escenarios se generan mediante
                **block bootstrap**, utilizando bloques de
                3 a 6 meses históricos.

                Todas las políticas utilizan exactamente
                los mismos escenarios.

                ### Transaction costs

                Cada trade tiene un costo de
                **10 basis points sobre el valor negociado**.

                ### Policies

                **Buy & Hold**  
                Mantiene el portafolio sin rebalancear.

                **Band Rebalancing (PFA)**  
                Rebalancea cuando los pesos se alejan más de
                una banda determinada del portafolio objetivo.

                **Lookahead (DLA)**  
                Cada 12 meses compara distintos niveles de exposición
                utilizando escenarios futuros simulados.

                **Panic Selling**  
                Sale del mercado después de un drawdown definido
                por el perfil del cliente.

                **Performance Chasing**  
                Mueve parte del portafolio hacia los activos que
                tuvieron mejor desempeño recientemente.

                ### Evaluación

                El tuning y la evaluación final utilizan
                semillas aleatorias diferentes.

                Para cada estrategia se reportan:

                - riqueza media;
                - riqueza mediana;
                - percentiles 5 y 95;
                - intervalo de confianza al 95%;
                - drawdown promedio;
                - costos de transacción.
                """
            )