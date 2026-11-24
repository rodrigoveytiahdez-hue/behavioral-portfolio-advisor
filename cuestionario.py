# cuestionario.py

import math

PREGUNTAS = [
    {
        "id": "q1",
        "titulo": "Inversión 1",
        "texto": "Tienes 50% de probabilidad de ganar $10,000 y 50% de perder $5,000. ¿Aceptarías la inversión?",
        "opciones": ["Sí", "No"],
    },
    {
        "id": "q2",
        "titulo": "Inversión 2",
        "texto": "Tienes 50% de probabilidad de ganar $15,000 y 50% de perder $5,000. ¿Aceptarías la inversión?",
        "opciones": ["Sí", "No"],
    },
    {
        "id": "q3",
        "titulo": "Inversión 3",
        "texto": "Tienes 50% de probabilidad de ganar $20,000 y 50% de perder $5,000. ¿Aceptarías la inversión?",
        "opciones": ["Sí", "No"],
    },
    {
        "id": "q4",
        "titulo": "Equivalente cierto — ganancia 1",
        "texto": "Tienes un boleto con 50% de probabilidad de ganar $10,000 y 50% de ganar $0. ¿Cuál es la cantidad segura más baja por la que venderías el boleto?",
        "opciones": [
            "$2,500",
            "$3,500",
            "$4,500",
            "$5,000",
            "$6,000",
        ],
    },
    {
        "id": "q5",
        "titulo": "Equivalente cierto — ganancia 2",
        "texto": "Tienes un boleto con 50% de probabilidad de ganar $20,000 y 50% de ganar $0. ¿Cuál es la cantidad segura más baja por la que venderías el boleto?",
        "opciones": [
            "$5,000",
            "$7,000",
            "$9,000",
            "$10,000",
            "$12,000",
        ],
    },
    {
        "id": "q6",
        "titulo": "Equivalente cierto — pérdida",
        "texto": "Enfrentas una situación con 50% de probabilidad de perder $10,000 y 50% de perder $0. ¿Cuál es la pérdida segura más alta que pagarías para evitar ese riesgo?",
        "opciones": [
            "$2,500",
            "$3,500",
            "$4,500",
            "$5,000",
            "$6,000",
        ],
    },
    {
        "id": "q7",
        "titulo": "Caída del portafolio",
        "texto": "Tu portafolio cae 20% en tres meses. ¿Qué harías?",
        "opciones": [
            "Vendería todo",
            "Vendería una parte",
            "No haría cambios",
            "Compraría más",
        ],
    },
    {
        "id": "q8",
        "titulo": "Mercado a la baja",
        "texto": "Tu portafolio lleva varios meses cayendo, pero tu plan de largo plazo no ha cambiado. ¿Qué harías?",
        "opciones": [
            "Saldría completamente del mercado",
            "Reduciría parte de mis inversiones",
            "Mantendría mi portafolio",
            "Aprovecharía para invertir más",
        ],
    },
    {
        "id": "q9",
        "titulo": "Activo con gran rendimiento",
        "texto": "Un sector del mercado subió 35% durante el último año. ¿Qué harías?",
        "opciones": [
            "Movería una parte importante de mi dinero a ese sector",
            "Aumentaría un poco mi inversión en ese sector",
            "Mantendría mi estrategia actual",
        ],
    },
    {
        "id": "q10",
        "titulo": "Activo con mal rendimiento",
        "texto": "Una de tus inversiones tuvo el peor rendimiento de tu portafolio durante el último año. ¿Qué harías?",
        "opciones": [
            "La vendería inmediatamente",
            "Reduciría mi posición",
            "La mantendría",
            "Revisaría sus fundamentos antes de tomar una decisión",
        ],
    },
]


def _nivel(score, bajo, alto):
    if score <= bajo:
        return "Bajo"
    elif score <= alto:
        return "Moderado"
    return "Alto"


def calcular_lambda(respuestas):
    """
    Aproximación simplificada de aversión a pérdidas.

    Las tres primeras preguntas aumentan la posible ganancia
    manteniendo fija la pérdida en $5,000.
    """

    aceptadas = [
        respuestas["q1"] == "Sí",
        respuestas["q2"] == "Sí",
        respuestas["q3"] == "Sí",
    ]

    # Buscamos la primera apuesta que el cliente acepta.
    if aceptadas[0]:
        return 1.5
    elif aceptadas[1]:
        return 2.5
    elif aceptadas[2]:
        return 3.5
    else:
        return 4.5


def parametro_desde_equivalente(equivalente, premio, minimo=0.30, maximo=1.20):
    """
    Con la función de valor v(x) = x^a y una lotería 50/50
    entre "premio" y 0, el equivalente cierto (CE) cumple:

        CE^a = 0.5 * premio^a   ->   a = ln(0.5) / ln(CE / premio)

    Si CE = premio/2 el cliente es neutral al riesgo (a = 1).
    """

    razon = equivalente / premio

    # Evitamos log(0) y divisiones entre cero
    razon = min(max(razon, 0.01), 0.99)

    valor = math.log(0.5) / math.log(razon)

    return min(max(valor, minimo), maximo)


def a_numero(texto):
    """
    Convierte "$4,500" en 4500.
    """

    return float(
        texto.replace("$", "").replace(",", "")
    )


def calcular_alpha(respuestas):
    """
    Alpha representa la curvatura frente a ganancias.
    Se estima con los equivalentes ciertos de q4 y q5
    y se promedian ambas estimaciones.

    alpha < 1 -> aversión al riesgo en ganancias.
    """

    alpha_q4 = parametro_desde_equivalente(
        a_numero(respuestas["q4"]),
        10_000
    )

    alpha_q5 = parametro_desde_equivalente(
        a_numero(respuestas["q5"]),
        20_000
    )

    return round((alpha_q4 + alpha_q5) / 2, 2)


def calcular_beta(respuestas):
    """
    Beta representa la curvatura frente a pérdidas.
    Con v(x) = -lambda * (-x)^beta y una lotería 50/50 entre
    perder $10,000 y $0, la pérdida segura máxima que el
    cliente pagaría cumple la misma fórmula que en ganancias
    (lambda se cancela).

    beta < 1 -> el cliente prefiere arriesgarse para evitar
    una pérdida segura (búsqueda de riesgo en pérdidas).
    """

    return round(
        parametro_desde_equivalente(
            a_numero(respuestas["q6"]),
            10_000
        ),
        2
    )


def calcular_panic_score(respuestas):
    score = 0

    mapa_q7 = {
        "Vendería todo": 3,
        "Vendería una parte": 2,
        "No haría cambios": 0,
        "Compraría más": 0,
    }

    mapa_q8 = {
        "Saldría completamente del mercado": 3,
        "Reduciría parte de mis inversiones": 2,
        "Mantendría mi portafolio": 0,
        "Aprovecharía para invertir más": 0,
    }

    score += mapa_q7[respuestas["q7"]]
    score += mapa_q8[respuestas["q8"]]

    return score


def calcular_chasing_score(respuestas):
    score = 0

    mapa_q9 = {
        "Movería una parte importante de mi dinero a ese sector": 3,
        "Aumentaría un poco mi inversión en ese sector": 1,
        "Mantendría mi estrategia actual": 0,
    }

    mapa_q10 = {
        "La vendería inmediatamente": 2,
        "Reduciría mi posición": 1,
        "La mantendría": 0,
        "Revisaría sus fundamentos antes de tomar una decisión": 0,
    }

    score += mapa_q9[respuestas["q9"]]
    score += mapa_q10[respuestas["q10"]]

    return score


def clasificar_perfil(lam, alpha, panic_score):
    """
    Clasificación sencilla para mostrar al cliente.
    Después la conectaremos con el modelo de optimización.
    """

    puntos = 0

    if lam <= 2:
        puntos += 2
    elif lam <= 3:
        puntos += 1

    if alpha >= 0.90:
        puntos += 2
    elif alpha >= 0.80:
        puntos += 1

    if panic_score <= 1:
        puntos += 2
    elif panic_score <= 3:
        puntos += 1

    if puntos >= 5:
        return "Crecimiento"
    elif puntos >= 3:
        return "Moderado"

    return "Conservador"


def calcular_perfil(respuestas):
    lam = calcular_lambda(respuestas)
    alpha = calcular_alpha(respuestas)
    beta = calcular_beta(respuestas)

    panic_score = calcular_panic_score(respuestas)
    chasing_score = calcular_chasing_score(respuestas)

    perfil = clasificar_perfil(
        lam,
        alpha,
        panic_score
    )

    return {
        "lambda": lam,
        "alpha": alpha,
        "beta": beta,
        "panic_score": panic_score,
        "chasing_score": chasing_score,
        "panic_level": _nivel(panic_score, 1, 3),
        "chasing_level": _nivel(chasing_score, 1, 3),
        "perfil": perfil,
    }