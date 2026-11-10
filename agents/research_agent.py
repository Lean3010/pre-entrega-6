import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from state import OrchestratorState

load_dotenv()


@tool
def search_knowledge_base(query: str) -> str:
    """
    Consulta la base de datos vectorial / fuentes externas sobre métricas de empresas,
    inversiones en IA, costos operativos y reseñas de clientes.
    """
    db_simulada = {
        "nvidia": (
            "Reporte Financiero Q3: Ingresos trimestrales de $35,100 millones USD. "
            "Costos operativos y de I+D: $12,400 millones USD. "
            "Reseñas de clientes Enterprise: 'El rendimiento en entrenamiento de LLMs es insuperable y excelente, "
            "aunque los tiempos de espera de stock generan frustración y los precios son muy elevados'."
        ),
        "mercado libre": (
            "Reporte Financiero Q3: Ingresos trimestrales de $5,300 millones USD. "
            "Costos operativos y logísticos: $3,850 millones USD. "
            "Reseñas de usuarios: 'Envíos rápidos en 24hs y excelente ecosistema fintech con Mercado Pago, "
            "pero las comisiones para vendedores subieron demasiado este trimestre'."
        ),
        "globant": (
            "Reporte Financiero Q3: Ingresos trimestrales de $615 millones USD. "
            "Costos operativos: $490 millones USD. "
            "Reseñas de clientes: 'Gran capacidad técnica en integración de agentes de IA y excelente soporte, "
            "aunque los costos de consultoría senior son altos'."
        ),
    }

    query_lower = query.lower()
    resultados = [
         f"[{clave.upper()}]: {valor}"
        for clave, valor in db_simulada.items()
        if clave in query_lower
    ]

    if not resultados:
        # Si la búsqueda es general, devuelve el panorama completo del sector tecnológico
        return "\n".join([f"[{k.upper()}]: {v}" for k, v in db_simulada.items()])

    return "\n\n".join(resultados)


def research_node(state: OrchestratorState) -> dict:
    """
    Nodo del Agente Investigador.
    Recibe una instrucción acotada para evitar contaminación de contexto.
    """
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        temperature=0
    )

    system_prompt = (
        "Eres un Agente Especialista en Investigación Documental y Búsqueda.\n"
        "Tu único objetivo es usar la herramienta 'search_knowledge_base' para recolectar "
        "datos duros (ingresos, costos y citas textuales de reseñas de clientes).\n"
        "REGLA ESTRICTA: No realices cálculos de rentabilidad ni análisis de sentimiento por tu cuenta; "
        "limítate a extraer y estructurar los datos crudos encontrados."
    )

    agent = create_react_agent(llm, tools=[search_knowledge_base], prompt=system_prompt)

    # Aislamiento de contexto: le pasamos solo la orden puntual del supervisor + consulta raíz
    user_request = state["messages"][0].content
    instruction = state.get("current_instruction") or f"Busca datos relevantes para: {user_request}"

    print(f"\n🔍 [RESEARCH AGENT] Ejecutando búsqueda -> '{instruction}'")

    result = agent.invoke({"messages": [HumanMessage(content=instruction)]})
    output_text = result["messages"][-1].content

    return {
        "research_data": output_text,
        "messages": [
            AIMessage(
                content=f"[Reporte de Investigación]:\n{output_text}",
                name="research_agent"
            )
        ],
    }