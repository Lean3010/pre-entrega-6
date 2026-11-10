import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from state import OrchestratorState

load_dotenv()


@tool
def calculate_financial_metrics(revenue_musd: float, cost_musd: float) -> dict:
    """
    Calcula la ganancia neta, el margen neto porcentual y el ROI a partir de los ingresos
    (revenue_musd) y costos (cost_musd) expresados en millones de USD.
    """
    if revenue_musd <= 0 or cost_musd <= 0:
        return {"error": "Los ingresos y costos deben ser mayores a 0."}

    net_profit = round(revenue_musd - cost_musd, 2)
    net_margin_pct = round((net_profit / revenue_musd) * 100, 2)
    roi_pct = round((net_profit / cost_musd) * 100, 2)

    return {
        "ganancia_neta_musd": net_profit,
        "margen_neto_porcentaje": f"{net_margin_pct}%",
        "roi_porcentaje": f"{roi_pct}%",
        "estado_esquema": "VALIDADO_OK"
    }


@tool
def analyze_customer_sentiment(review_text: str) -> dict:
    """
    Analiza el sentimiento y extrae polaridad cuantitativa de las reseñas de clientes.
    """
    positivas = ["insuperable", "excelente", "rápidos", "gran capacidad", "sólido"]
    negativas = ["frustración", "elevados", "subieron demasiado", "altos", "espera"]

    texto = review_text.lower()
    pos_hits = sum(1 for p in positivas if p in texto)
    neg_hits = sum(1 for n in negativas if n in texto)

    total = pos_hits + neg_hits
    score = round((pos_hits - neg_hits) / max(total, 1), 2)

    if score > 0.15:
        etiqueta = "POSITIVO"
    elif score < -0.15:
        etiqueta = "NEGATIVO"
    else:
        etiqueta = "MIXTO / NEUTRAL"

    return {
        "sentimiento_global": etiqueta,
        "score_polaridad": score,
        "aspectos_positivos_detectados": pos_hits,
        "puntos_de_dolor_detectados": neg_hits,
    }


def analyst_node(state: OrchestratorState) -> dict:
    """
    Nodo del Agente Analista/Cómputo.
    Toma únicamente los datos recolectados en 'research_data' y aplica herramientas cuantitativas.
    """
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        temperature=0
    )

    system_prompt = (
        "Eres un Agente Especialista en Análisis Financiero, Cómputo y Sentimiento.\n"
        "Debes utilizar OBLIGATORIAMENTE tus herramientas 'calculate_financial_metrics' y "
        "'analyze_customer_sentiment' sobre los datos crudos provistos para obtener métricas exactas.\n"
        "Nunca inventes números ni hagas cuentas mentales sin invocar la herramienta."
    )

    agent = create_react_agent(
        llm,
        tools=[calculate_financial_metrics, analyze_customer_sentiment],
        prompt=system_prompt
    )

    datos_investigacion = state.get("research_data", "No hay datos previos.")
    instruccion = state.get("current_instruction", "Procesa los datos financieros y de sentimiento.")

    # Contexto acotado: solo recibe la orden del supervisor y el output limpio del investigador
    prompt_acotado = (
        f"Instrucción del Supervisor: {instruccion}\n\n"
        f"Datos crudos del Agente Investigador:\n{datos_investigacion}"
    )

    print(f"\n📊 [ANALYST AGENT] Procesando métricas y sentimiento -> '{instruccion}'")

    result = agent.invoke({"messages": [HumanMessage(content=prompt_acotado)]})
    output_text = result["messages"][-1].content

    return {
        "analysis_data": output_text,
        "messages": [
            AIMessage(
                content=f"[Reporte de Análisis y Cómputo]:\n{output_text}",
                name="analyst_agent"
            )
        ],
    }