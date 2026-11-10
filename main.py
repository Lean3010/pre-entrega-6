import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from state import OrchestratorState
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node

load_dotenv()

MAX_ITERATIONS = 5  # Límite estricto contra el "Supervisor Infinito"


# Esquema Pydantic para forzar decisión estructurada y validación del Supervisor
class SupervisorDecision(BaseModel):
    next_agent: Literal["research_agent", "analyst_agent", "FINISH"] = Field(
        description="Agente que debe intervenir ahora o 'FINISH' si la tarea cumplió toda la rúbrica."
    )
    instruction_for_agent: str = Field(
        description="Instrucción puntual y acotada para el especialista elegido (vacío si es FINISH)."
    )
    validation_reasoning: str = Field(
        description="Evaluación de suficiencia y calidad sobre los datos actuales."
    )
    final_synthesis: str = Field(
        description="Síntesis ejecutiva final integrando investigación y análisis. Obligatorio cuando next_agent es FINISH."
    )


def supervisor_node(state: OrchestratorState) -> dict:
    """
    Nodo Supervisor Jerárquico: evalúa el estado estructurado, valida la rúbrica de calidad
    y decide dinámicamente qué especialista actúa o si finaliza sintetizando la respuesta.
    """
    iteration = state.get("iteration_count", 0) + 1
    research_data = state.get("research_data")
    analysis_data = state.get("analysis_data")
    user_query = state["messages"][0].content

    # Salvaguarda dura contra el "Supervisor Infinito"
    if iteration > MAX_ITERATIONS:
        print("\n⚠️ [SUPERVISOR] Límite máximo de iteraciones alcanzado. Forzando cierre seguro.")
        return {
            "next_agent": "FINISH",
            "iteration_count": iteration,
            "final_report": (
                f"Síntesis de emergencia (límite de pasos alcanzado):\n\n"
                f"Investigación:\n{research_data}\n\nAnálisis:\n{analysis_data}"
            ),
        }

    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        temperature=0
    )
    structured_supervisor = llm.with_structured_output(SupervisorDecision)

    system_prompt = (
        "Eres el Supervisor Jerárquico de un Orquestador Multi-Agente de Investigación y Análisis.\n"
        "Dada la solicitud del usuario y el estado actual, debes decidir quién actúa ahora o si es momento de finalizar.\n\n"
        "RÚBRICA DE VALIDACIÓN ESTRICTA ANTES DE MARCAR 'FINISH':\n"
        "1. Si 'research_data' está vacío o incompleto -> Enruta a 'research_agent' indicando qué buscar.\n"
        "2. Si 'research_data' ya tiene datos crudos pero 'analysis_data' está vacío o le faltan cálculos de ROI/Margen/Sentimiento -> Enruta a 'analyst_agent'.\n"
        "3. Si AMBOS ('research_data' y 'analysis_data') están completos y validados -> Elige 'FINISH' y redacta en 'final_synthesis' "
        "un informe ejecutivo final claro y bien estructurado para el usuario."
    )

    state_summary = (
        f"Solicitud original del usuario: {user_query}\n"
        f"Iteración actual: {iteration}/{MAX_ITERATIONS}\n"
        f"¿Datos de Investigación disponibles?: {'SÍ -> ' + research_data if research_data else 'NO (Pendiente)'}\n"
        f"¿Datos de Análisis/Cómputo disponibles?: {'SÍ -> ' + analysis_data if analysis_data else 'NO (Pendiente)'}"
    )

    decision: SupervisorDecision = structured_supervisor.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state_summary),
    ])

    print(f"\n🧠 [SUPERVISOR - Paso {iteration}] Decisión: {decision.next_agent}")
    print(f"   📋 Validación: {decision.validation_reasoning}")

    updates = {
        "next_agent": decision.next_agent,
        "current_instruction": decision.instruction_for_agent,
        "iteration_count": iteration,
    }

    if decision.next_agent == "FINISH":
        updates["final_report"] = decision.final_synthesis
        updates["messages"] = [
            AIMessage(content=decision.final_synthesis, name="supervisor")
        ]

    return updates


def route_from_supervisor(
    state: OrchestratorState,
) -> Literal["research_agent", "analyst_agent", "__end__"]:
    """Función de enrutamiento con retorno Literal para las Conditional Edges."""
    proximo = state.get("next_agent")
    if proximo == "research_agent":
        return "research_agent"
    if proximo == "analyst_agent":
        return "analyst_agent"
    return "__end__"


def build_graph():
    """Construye y compila el StateGraph con topología jerárquica."""
    workflow = StateGraph(OrchestratorState)

    # 1. Registro de Nodos
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("analyst_agent", analyst_node)

    # 2. Punto de entrada al Supervisor
    workflow.add_edge(START, "supervisor")

    # 3. Aristas Condicionales desde el Supervisor hacia los Especialistas o END
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "__end__": END,
        },
    )

    # 4. Retorno obligatorio de los Especialistas al Supervisor para validación
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    # Imprime el diagrama Mermaid en consola por si quieres verificarlo
    print("=== DIAGRAMA MERMAID DEL GRAFO ===")
    print(app.get_graph().draw_mermaid())
    print("==================================\n")

    consulta_usuario = (
        "Necesito un informe completo que compare a NVIDIA y Mercado Libre en el Q3: "
        "busca sus ingresos, costos y reseñas de clientes; luego calcula el margen neto, "
        "el ROI de cada una y analiza el sentimiento de sus usuarios para recomendar cuál tiene mejor retorno."
    )

    print(f"🚀 Iniciando Orquestador con consulta:\n'{consulta_usuario}'")

    estado_inicial = {
        "messages": [HumanMessage(content=consulta_usuario)],
        "research_data": None,
        "analysis_data": None,
        "final_report": None,
        "current_instruction": "",
        "iteration_count": 0,
    }

    resultado_final = app.invoke(estado_inicial)

    print("\n" + "=" * 65)
    print("✅ SÍNTESIS FINAL DEL ORQUESTADOR MULTI-AGENTE")
    print("=" * 65)
    print(resultado_final["final_report"])