"""
Local vector store implementation.

This module defines a ``VectorStore`` class that wraps a simple ChromaDB
collection.  It provides methods to upsert vectors, query nearest neighbours
and delete vectors.  All data is persisted locally to the directory
specified at initialisation time.  Before using this class ensure that the
``chromadb`` package is installed and that a supported storage backend
is available (the default uses DuckDB on disk).
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings


class VectorStore:
    """A minimal local vector store using ChromaDB.

    Parameters
    ----------
    persist_directory : str, optional
        Directory on disk where ChromaDB should persist its data.  If not
        provided, a default directory named ``local_chroma`` will be used.

    collection_name : str, optional
        Name of the collection to use.  Defaults to ``default``.
    """

    def __init__(
        self,
        persist_directory: str = "local_chroma",
        collection_name: str = "default",
    ) -> None:
        # Configure persistent storage
        settings = Settings(persist_directory=persist_directory)
        self._client = chromadb.Client(settings)
        # Create or load the collection
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        ids: Iterable[str],
        embeddings: Iterable[List[float]],
        metadata: Optional[Iterable[Dict[str, object]]] = None,
    ) -> None:
        """Insert or update vectors with optional metadata.

        Parameters
        ----------
        ids : Iterable[str]
            Unique identifiers for each vector.
        embeddings : Iterable[List[float]]
            List of embedding vectors.
        metadata : Iterable[Dict[str, object]] or None
            Optional metadata for each vector.  If not provided, empty dicts are
            used.
        """
        metadatas = list(metadata) if metadata is not None else [{} for _ in ids]
        self._collection.upsert(
            ids=list(ids), embeddings=list(embeddings), metadatas=metadatas
        )

    def query(
        self,
        embedding: List[float],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the IDs and distances of the ``k`` nearest neighbours.

        Parameters
        ----------
        embedding : List[float]
            The query embedding.
        k : int, optional
            Number of results to return.  Defaults to ``5``.

        Returns
        -------
        List[Tuple[str, float]]
            Pairs of (id, distance) for the nearest neighbours.
        """
        results = self._collection.query(embeddings=[embedding], n_results=k)
        return list(zip(results["ids"][0], results["distances"][0]))

    def delete(self, ids: Iterable[str]) -> None:
        """Delete vectors from the store.

        Parameters
        ----------
        ids : Iterable[str]
            Unique identifiers of the vectors to remove.
        """
        self._collection.delete(ids=list(ids))


__all__ = ["VectorStore"]