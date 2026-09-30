"""Deterministic terminal nodes for routes that need no tool or model."""

from bank_assistant.state.schemas import AssistantState, Outcome, reply


def handoff(_state: AssistantState) -> dict:
    return reply(
        Outcome.HANDOFF,
        "Tu solicitud necesita la atención de un agente. Contacta con atención al cliente "
        "de lunes a sábado, de 8:00 a 22:00.",
    )


def out_of_scope(_state: AssistantState) -> dict:
    return reply(
        Outcome.OUT_OF_SCOPE,
        "Solo puedo ayudarte con información de Banco Aurora, simulaciones hipotecarias, "
        "consultas de tus cuentas y bloqueo de tarjetas.",
    )


def clarify(_state: AssistantState) -> dict:
    return reply(
        Outcome.NEEDS_INFO,
        "No he entendido bien tu solicitud. ¿Es una consulta general, una simulación "
        "hipotecaria, una consulta de tu cuenta o el bloqueo de una tarjeta?",
    )
