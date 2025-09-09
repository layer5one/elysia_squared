"""
Tool‑enabled agent for offline inference.

This module defines a ``ToolAgent`` class that extends the basic
retrieval‑augmented ``Agent`` with the ability to call local tools
during a conversation.  The agent advertises available tools to the
language model and interprets JSON tool call outputs.  When a tool
is invoked, its result is appended to the conversation and passed
back into the model for further reasoning.

The ``ToolAgent`` assumes the underlying language model can be
prompted to emit tool calls in the format:

```
{"name": "tool_name", "parameters": {"param1": "value", ...}}
```

If the model output parses as JSON with ``name`` and ``parameters``
fields and the ``name`` matches a registered tool, the agent
executes the tool and provides its output back to the model.  If
parsing fails, the model output is returned verbatim as the final
answer.

This implementation does not rely on any remote APIs and runs
entirely locally.  It can be extended with additional tools by
updating the ``TOOLS`` mapping in ``tools.py``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

from .agent import Agent
from .tools import TOOLS


class ToolAgent(Agent):
    """Agent with tool‑calling capability.

    Parameters
    ----------
    vector_store : VectorStore
        The underlying vector store for storing and retrieving embeddings.
    embedder : GemmaEmbedder
        Embedding model used to convert text into vectors.
    model : GemmaModel
        Language model used to generate answers and tool calls.
    top_k : int, optional
        Number of relevant documents to retrieve for each query.  Defaults
        to 3.
    max_tool_calls : int, optional
        Maximum number of tool invocations per user query to prevent
        infinite loops.  Defaults to 3.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        model,
        top_k: int = 3,
        max_tool_calls: int = 3,
    ) -> None:
        super().__init__(vector_store, embedder, model, top_k)
        self.max_tool_calls = max_tool_calls
        # Conversation history: list of dicts with keys 'role' and 'content'
        # Roles can be 'user', 'assistant' or 'tool'.  For tool messages,
        # an additional 'name' field holds the tool name.
        self.history: List[Dict[str, Any]] = []

    def _tools_description(self) -> str:
        """Construct a description of available tools for the model."""
        lines = []
        for name, meta in TOOLS.items():
            lines.append(f"- {name}: {meta['description']}")
        return "\n".join(lines)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt string for the language model.

        The prompt includes a system message describing the available
        tools, the conversation history, retrieved context and the
        latest user question.  The model is instructed to output
        tool calls as JSON objects when appropriate.
        """
        # System instruction
        sys_prompt = (
            "You are a helpful assistant with access to the following tools:\n"
            f"{self._tools_description()}\n\n"
            "When you determine that a tool is needed to answer the user, "
            "respond with a JSON object with exactly two keys: 'name' and "
            "'parameters'.  'name' should be the tool name, and 'parameters' "
            "should be a JSON object containing the arguments for the tool. "
            "If no tool is needed, simply answer the question in plain text."
        )
        parts: List[str] = [sys_prompt]
        # Append history
        for message in self.history:
            role = message.get("role")
            if role == "user":
                parts.append(f"User: {message['content']}")
            elif role == "assistant":
                parts.append(f"Assistant: {message['content']}")
            elif role == "tool":
                # Provide tool result back to model
                tool_name = message.get("name", "tool")
                parts.append(f"Tool ({tool_name}) result: {message['content']}")
        # Include retrieved context
        if context:
            parts.append(f"Context:\n{context}")
        # Current user question
        parts.append(f"User: {query}")
        # Prompt the assistant to respond
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def chat(self, query: str) -> str:
        """Process a user query, possibly invoking tools.

        This method handles multi‑turn interactions where the model may
        request tool invocations.  It iteratively feeds the model
        generated tool calls and their outputs until a final answer
        (non‑JSON) is produced or the maximum number of tool calls is
        reached.

        Parameters
        ----------
        query : str
            The user's question.

        Returns
        -------
        str
            The final answer from the model after executing any needed
            tools.
        """
        # Append the user message to the history
        self.history.append({"role": "user", "content": query})
        # Embed and retrieve context
        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.query(embedding=query_embedding, k=self.top_k)
        context_texts: List[str] = []
        for doc_id, _ in results:
            text = self._documents.get(doc_id)
            if text:
                context_texts.append(text)
        context = "\n\n".join(context_texts)
        # Iterate tool calls
        for _ in range(self.max_tool_calls + 1):
            prompt = self._build_prompt(query, context)
            model_output = self.model.generate(prompt)
            # Try to parse as JSON
            output_str = model_output.strip()
            try:
                tool_call = json.loads(output_str)
                # Expect a dict with 'name' and 'parameters'
                if (
                    isinstance(tool_call, dict)
                    and "name" in tool_call
                    and "parameters" in tool_call
                    and tool_call["name"] in TOOLS
                ):
                    tool_name = tool_call["name"]
                    params = tool_call.get("parameters", {})
                    # Execute tool
                    result = TOOLS[tool_name]["function"](**params)
                    # Append tool result to history
                    self.history.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    # After tool execution, continue the loop to get next assistant response
                    continue
                else:
                    # JSON parsed but not a valid tool call; treat as answer
                    self.history.append({"role": "assistant", "content": output_str})
                    return output_str
            except json.JSONDecodeError:
                # Not JSON; treat as final answer
                self.history.append({"role": "assistant", "content": output_str})
                return output_str
        # If we exit the loop, we hit max tool calls; return last response
        self.history.append({"role": "assistant", "content": output_str})
        return output_str


__all__ = ["ToolAgent"]