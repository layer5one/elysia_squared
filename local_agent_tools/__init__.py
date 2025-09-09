"""
Local Agent package

This package provides a lightweight offline agentic framework built around a
vector store, an embedding model and a local language model.  It is intended
to run entirely on your machine without any external services once the
necessary models have been downloaded.  See the ``main.py`` module for a
simple command‑line interface.
"""

from .vector_store import VectorStore
from .embedding import GemmaEmbedder
from .model import GemmaModel
from .agent import Agent
from .tool_agent import ToolAgent
from .tools import TOOLS

__all__ = [
    "VectorStore",
    "GemmaEmbedder",
    "GemmaModel",
    "Agent",
    "ToolAgent",
    "TOOLS",
]