"""
Embedding models for the local agent.

This module defines a ``GemmaEmbedder`` class that loads Google's
``embeddinggemma-300m`` model and exposes a simple interface for generating
vector representations of text.  The model is loaded via the ``transformers``
library and will attempt to download weights from Hugging Face the first
time it is used.  To run entirely offline, ensure that the model weights
are present in your local Hugging Face cache (see the Transformers
documentation for details) or specify a local path.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from transformers import AutoTokenizer, AutoModel
import torch


class GemmaEmbedder:
    """Wrapper around the ``google/embeddinggemma-300m`` text embedding model.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the embedding model.  Defaults to
        ``"google/embeddinggemma-300m"``.
    device : str or torch.device, optional
        Device on which to run inference (e.g. "cpu" or "cuda").  If not
        provided, will use ``cuda`` if available else ``cpu``.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._model = self._model.to(self._device)
        self._model.eval()

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple text strings into vectors.

        Parameters
        ----------
        texts : Iterable[str]
            The input texts to embed.

        Returns
        -------
        List[List[float]]
            A list of dense embedding vectors.  Each vector corresponds to
            one input text and is represented as a list of floats.
        """
        # Prepare inputs
        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Use the [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        return embeddings.tolist()


__all__ = ["GemmaEmbedder"]