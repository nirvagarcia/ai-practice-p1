"""Lab CLI. Composition and terminal I/O only; no domain logic."""

import argparse

from pydantic import ValidationError

from bank_assistant.application.bootstrap import build_assistant, build_faq
from bank_assistant.application.service import AssistantReply, AssistantService
from bank_assistant.config.settings import ConfigurationError, get_settings
from bank_assistant.domain.schemas import Principal
from bank_assistant.workflows.faq.baseline import FaqResult

RAG_SMOKE = (
    "¿Cuál es el horario de las oficinas los jueves?",
    "¿Ofrecéis seguros para mascotas?",
)
GRAPH_SMOKE = (
    "¿A qué hora abren las oficinas los jueves?",
    "Simula una hipoteca de 200.000 € al 3 % a 25 años",
    "¿Cuál es mi saldo?",
    "Me han robado la tarjeta, bloquéala",
    "Quiero poner una reclamación por un cobro indebido",
    "Escríbeme un poema sobre el mar",
)
EXIT_WORDS = {"salir", "exit", "quit"}


def render_faq(result: FaqResult) -> str:
    retrieval = result.retrieval
    hits = "\n".join(f"  {c.score:.3f}  {c.chunk_id}" for c in retrieval.chunks) or "  (none)"
    return (
        f"Q: {retrieval.query}\n"
        f"Retrieved in {retrieval.duration_ms:.0f} ms:\n{hits}\n"
        f"A: {result.answer}\n"
        f"Sources: {', '.join(result.source_ids) or '-'}\n"
    )


def render_reply(reply: AssistantReply) -> str:
    sources = f"\n  fuentes: {', '.join(reply.source_ids)}" if reply.source_ids else ""
    return f"[{reply.intent or '-'} → {reply.outcome}] {reply.text}{sources}\n"


def chat(service: AssistantService, principal: Principal) -> None:
    print(f"Banco Aurora (lab) — cliente {principal.customer_id}. Escribe 'salir' para terminar.")
    try:
        while (message := input("> ").strip()).lower() not in EXIT_WORDS:
            if message:
                print(render_reply(service.ask(message, principal)))
    except (EOFError, KeyboardInterrupt):
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bank_assistant")
    parser.add_argument("--mode", choices=["chat", "rag-smoke", "graph-smoke"], default="chat")
    parser.add_argument("--customer", default="CUST-0001", help="synthetic lab principal")
    parser.add_argument("--message", action="append", help="repeatable; overrides smoke set")
    args = parser.parse_args(argv)

    try:
        principal = Principal(customer_id=args.customer)
    except ValidationError:
        parser.error("--customer must look like CUST-0000")

    settings = get_settings()
    try:
        if args.mode == "rag-smoke":
            faq = build_faq(settings)
            for question in args.message or RAG_SMOKE:
                print(render_faq(faq.answer(question)))
            return
        service = build_assistant(settings)
    except ConfigurationError as error:
        parser.exit(2, f"{error}\n")

    if args.mode == "graph-smoke":
        for message in args.message or GRAPH_SMOKE:
            print(f"> {message}\n{render_reply(service.ask(message, principal))}")
    else:
        chat(service, principal)


if __name__ == "__main__":
    main()
