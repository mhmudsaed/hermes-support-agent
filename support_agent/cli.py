"""CLI: ask / ingest / tickets subcommands."""

from __future__ import annotations

import argparse
import json
import sys

from support_agent.config import load_settings
from support_agent.service import SupportAgent
from support_agent.telegram import TelegramBot, make_transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-agent",
        description="Bilingual support agent: cited answers from your docs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest docs into the knowledge base")
    p_ingest.add_argument("paths", nargs="+",
                          help="files or directories to ingest (.md/.txt)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question from the docs")
    p_ask.add_argument("question", help="the customer question (ar/en)")
    p_ask.add_argument("--contact", default="",
                       help="contact info stored if a ticket is opened")
    p_ask.add_argument("--json", action="store_true",
                       help="emit machine-readable JSON")
    p_ask.set_defaults(func=cmd_ask)

    p_tix = sub.add_parser("tickets", help="work with support tickets")
    tix_sub = p_tix.add_subparsers(dest="tickets_command", required=True)

    p_list = tix_sub.add_parser("list", help="list tickets")
    p_list.add_argument("--status", default="", help="filter: open/answered/closed")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_tickets_list)

    p_show = tix_sub.add_parser("show", help="show a ticket + conversation log")
    p_show.add_argument("ticket_id")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=cmd_tickets_show)

    p_reply = tix_sub.add_parser("reply", help="record a human reply (follow-up)")
    p_reply.add_argument("ticket_id")
    p_reply.add_argument("text", help="the human's reply")
    p_reply.add_argument("--close", action="store_true",
                         help="mark ticket answered")
    p_reply.set_defaults(func=cmd_tickets_reply)

    p_retry = tix_sub.add_parser("retry", help="re-run a ticket question vs the KB")
    p_retry.add_argument("ticket_id")
    p_retry.set_defaults(func=cmd_tickets_retry)

    p_serve = sub.add_parser("serve", help="run the Telegram bot loop")
    p_serve.add_argument("--once", action="store_true",
                         help="handle one poll batch then exit")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def _agent() -> SupportAgent:
    return SupportAgent(load_settings())


def cmd_ingest(args: argparse.Namespace) -> int:
    import glob
    import os

    agent = _agent()
    files, chunks = 0, 0
    for raw in args.paths:
        if os.path.isdir(raw):
            result = agent.ingest_dir(raw)
            files += result["files"]
            chunks += result["chunks"]
        else:
            for path in sorted(glob.glob(raw)) or [raw]:
                chunks += agent.ingest_path(path)
                files += 1
    print(f"ingested {files} file(s), {chunks} chunk(s)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    agent = _agent()
    try:
        result = agent.ask(args.question, contact=args.contact)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"kind": result.kind, "text": result.text,
                          "citations": result.citations,
                          "confidence": round(result.confidence, 4),
                          "ticket_id": result.ticket.id if result.ticket else None},
                         ensure_ascii=False, indent=2))
    else:
        print(result.text)
        if result.ticket is not None and result.kind == "ticket":
            print(f"[ticket {result.ticket.id}]")
    return 0


def _ticket_dict(ticket, messages=None) -> dict:
    d = {"id": ticket.id, "question": ticket.question,
         "summary": ticket.summary, "contact": "***" if ticket.contact else "",
         "status": ticket.status, "candidates": ticket.candidates,
         "created_at": ticket.created_at}
    if messages is not None:
        d["messages"] = messages
    return d


def cmd_tickets_list(args: argparse.Namespace) -> int:
    agent = _agent()
    tickets = agent.list_tickets(args.status)
    if args.json:
        print(json.dumps([_ticket_dict(t) for t in tickets],
                         ensure_ascii=False, indent=2))
    elif not tickets:
        print("No tickets.")
    else:
        for t in tickets:
            print(f"{t.id} [{t.status}] {t.question[:90]}")
    return 0


def cmd_tickets_show(args: argparse.Namespace) -> int:
    agent = _agent()
    try:
        ticket, messages = agent.ticket_log(args.ticket_id)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_ticket_dict(ticket, messages),
                         ensure_ascii=False, indent=2))
    else:
        print(f"Ticket {ticket.id} [{ticket.status}]")
        print(f"Q: {ticket.question}")
        print("--- conversation ---")
        for m in messages:
            print(f"[{m['role']}] {m['text']}")
    return 0


def cmd_tickets_reply(args: argparse.Namespace) -> int:
    agent = _agent()
    try:
        ticket = agent.reply_ticket(args.ticket_id, args.text, close=args.close)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"reply recorded on {ticket.id} (status: {ticket.status})")
    return 0


def cmd_tickets_retry(args: argparse.Namespace) -> int:
    agent = _agent()
    try:
        result = agent.try_answer_from_context(args.ticket_id)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    settings = load_settings()
    agent = SupportAgent(settings)
    transport = make_transport(settings)
    bot = TelegramBot(agent=agent, transport=transport)
    if args.once:
        print(f"handled {bot.serve_once()} message(s)")
    else:
        bot.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
