"""Card block request. Until HITL exists, it never executes the action: it hands off."""

from bank_assistant.state.schemas import AssistantState, Outcome, reply

PENDING_APPROVAL_NOTICE = (
    "El bloqueo de tarjetas requiere aprobación de un agente y aún no se ejecuta desde este "
    "asistente: no se ha realizado ninguna acción. Si te han robado la tarjeta, llama a la "
    "línea de emergencias, disponible 24 horas."
)


def request_card_block(_state: AssistantState) -> dict:
    return reply(Outcome.HANDOFF, PENDING_APPROVAL_NOTICE)
