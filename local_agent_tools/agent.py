"""
Agent implementation for offline inference.

The ``Agent`` class brings together the embedding model, vector store and
language model to provide a simple Retrieval‑Augmented Generation (RAG) flow.
You can ingest documents into the agent's vector store and then answer
queries by retrieving the most relevant documents and prompting the
language model with that context.  This implementation runs entirely
locally once the models and embeddings are cached and stored on disk.
"""

from __future__ import annotations

from typing import List, Iterable, Optional

from .vector_store import VectorStore
from .embedding import GemmaEmbedder
from .model import GemmaModel


class Agent:
    """A simple retrieval‑augmented agent for offline question answering.

    Parameters
    ----------
    vector_store : VectorStore
        The underlying vector store for storing and retrieving embeddings.
    embedder : GemmaEmbedder
        Embedding model used to convert text into vectors.
    model : GemmaModel
        Language model used to generate answers.
    top_k : int, optional
        Number of relevant documents to retrieve for each query.  Defaults to 3.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: GemmaEmbedder,
        model: GemmaModel,
        top_k: int = 3,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self.top_k = top_k
        # Store full text of ingested documents keyed by their IDs
        self._documents: dict[str, str] = {}

    def ingest(self, doc_id: str, text: str) -> None:
        """Ingest a document into the vector store.

        Parameters
        ----------
        doc_id : str
            Unique identifier for the document.
        text : str
            The full text of the document.  It will be converted to an embedding
            and stored in the vector store.
        """
        # Compute embedding for the document
        embedding = self.embedder.embed([text])[0]
        # Upsert into vector store
        self.vector_store.upsert(ids=[doc_id], embeddings=[embedding], metadata=[{}])
        # Save the raw text for later retrieval
        self._documents[doc_id] = text

    def answer(self, query: str) -> str:
        """Answer a user query using retrieved context and the language model.

        Parameters
        ----------
        query : str
            The user's question.

        Returns
        -------
        str
            The generated answer.
        """
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]
        # Retrieve top_k relevant docs
        results = self.vector_store.query(embedding=query_embedding, k=self.top_k)
        context_texts: List[str] = []
        for doc_id, distance in results:
            text = self._documents.get(doc_id)
            if text is not None:
                context_texts.append(text)
        # Build prompt: include retrieved context and the question
        context_section = "\n\n".join(context_texts)
        prompt = (
            "You are a helpful assistant.\n\n"
            f"Context:\n{context_section}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        # Generate answer
        answer = self.model.generate(prompt)
        return answer.strip()


__all__ = ["Agent"]