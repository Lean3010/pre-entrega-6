from typing import Literal, Optional
from langgraph.graph import MessagesState


class OrchestratorState(MessagesState):
    """
    Estado compartido estructurado del Orquestador Multi-Agente.
    Hereda 'messages' de MessagesState y añade trazabilidad por especialista.
    """
    # Próximo nodo a ejecutar decidido por el Supervisor
    next_agent: Literal["research_agent", "analyst_agent", "FINISH"]
    
    # Instrucción acotada que el Supervisor le envía al especialista activo
    current_instruction: str
    
    # Almacenamiento estructurado por dominio para evitar pérdida de contexto
    research_data: Optional[str]
    analysis_data: Optional[str]
    
    # Síntesis final validada para entregar al usuario
    final_report: Optional[str]
    
    # Contador de seguridad contra el "Supervisor Infinito"
    iteration_count: int