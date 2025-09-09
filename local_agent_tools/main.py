"""
Command‑line interface for the local agent.

This script brings together the vector store, embedder and language model
to form a simple offline question answering system.  You can ingest
documents (from files or standard input) into the vector store and then
answer questions using those documents as context.
"""

from __future__ import annotations

import argparse
import os
import sys

from .vector_store import VectorStore
from .embedding import GemmaEmbedder
from .model import GemmaModel
from .agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local agent offline interface")
    parser.add_argument(
        "--store-dir",
        type=str,
        default="local_chroma",
        help="Directory to persist the vector store (default: local_chroma)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Name of the collection to use in the vector store.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="google/embeddinggemma-300m",
        help="Embedding model name or path (default: google/embeddinggemma-300m)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the local language model (e.g., google/gemma-3-12b-it-qat)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for running models (cpu, cuda) (default: auto)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve for each query (default: 3)",
    )
    parser.add_argument(
        "--tool-agent",
        action="store_true",
        help="Enable tool‑calling agent mode.  In this mode the assistant can invoke local tools such as reading files or running Python code."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Initialise components
    store = VectorStore(persist_directory=args.store_dir, collection_name=args.collection)
    embedder = GemmaEmbedder(model_name=args.embedder, device=args.device)
    model = GemmaModel(model_name=args.model, device=args.device)
    if args.tool_agent:
        from .tool_agent import ToolAgent
        agent = ToolAgent(
            vector_store=store,
            embedder=embedder,
            model=model,
            top_k=args.top_k,
        )
    else:
        agent = Agent(
            vector_store=store,
            embedder=embedder,
            model=model,
            top_k=args.top_k,
        )

    if args.tool_agent:
        print("Tool‑enabled agent ready. Type 'help' for commands.")
    else:
        print("Local agent ready. Type 'help' for commands.")
    while True:
        try:
            command = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not command:
            continue
        if command.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        if command.lower() == "help":
            print("Commands:")
            print("  ingest <id> <path>   - ingest a text file into the vector store")
            if args.tool_agent:
                print("  chat <question>     - ask a question; the assistant may invoke tools")
            else:
                print("  query <question>     - ask a question")
            print("  exit | quit          - exit the program")
            continue
        parts = command.split(maxsplit=2)
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "ingest":
            if len(parts) < 3:
                print("Usage: ingest <id> <path>")
                continue
            doc_id = parts[1]
            path = parts[2]
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            agent.ingest(doc_id, text)
            print(f"Ingested document {doc_id} ({len(text.split())} words)")
        elif cmd == "query" and not args.tool_agent:
            # simple query mode
            if len(parts) < 2:
                print("Usage: query <question>")
                continue
            question = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
            answer = agent.answer(question)
            print("Answer:\n", answer)
        elif cmd == "chat" and args.tool_agent:
            # tool‑enabled chat mode
            if len(parts) < 2:
                print("Usage: chat <question>")
                continue
            question = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
            answer = agent.chat(question)
            print("Assistant:\n", answer)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for a list of commands.")


if __name__ == "__main__":
    main()