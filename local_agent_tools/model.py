"""
Local language model wrapper for the agent.

This module provides a ``GemmaModel`` class that loads a quantized
instruction‑tuned Gemma 3 model via the Transformers library.  It exposes
a simple ``generate`` method for producing completions given a prompt and
context.  The class assumes that the model weights have been downloaded
and cached locally; quantized versions (e.g. int4) are recommended for
running on consumer GPUs.  See Google's Gemma 3 documentation for
hardware requirements【68859278841765†L240-L272】.
"""

from __future__ import annotations

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GemmaModel:
    """Wrapper for a Gemma 3 instruction‑tuned language model.

    Parameters
    ----------
    model_name : str
        Name or path of the Gemma model.  For example:
        ``"google/gemma-3-12b-it-qat"`` or ``"orieg/gemma3-tools:12b-it-qat"``.
    device : str or torch.device, optional
        Device on which to run the model (``"cpu"`` or ``"cuda"``).  Defaults to
        ``"cuda"`` if available else ``"cpu"``.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._model = self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_token: Optional[str] = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            The input prompt (context and query) for the language model.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate.  Defaults to ``256``.
        temperature : float, optional
            Sampling temperature for generation.  Higher values produce more
            diverse outputs.  Defaults to ``0.7``.
        top_p : float, optional
            Nucleus sampling parameter.  Only the top ``p`` probability mass is
            considered.  Defaults to ``0.95``.
        stop_token : str or None, optional
            If provided, generation will stop when this token sequence is
            generated.  Defaults to ``None`` (no explicit stop).

        Returns
        -------
        str
            The generated completion text.
        """
        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        output = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if stop_token is not None and stop_token in output:
            output = output.split(stop_token)[0]
        # Remove the original prompt from the output to return only the new text
        if output.startswith(prompt):
            return output[len(prompt) :].lstrip()
        return output


__all__ = ["GemmaModel"]