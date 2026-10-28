from __future__ import annotations

from collections.abc import Iterable


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder:
    """Lazy local embedding implementation, independent of generation providers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load local embedding model {self.model_name}. "
                    "Check network access for the initial model download or configure a supported local model."
                ) from exc
        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        try:
            return [vector.tolist() for vector in self._get_model().embed(inputs)]
        except Exception as exc:
            raise RuntimeError(f"Local embedding failed: {type(exc).__name__}.") from exc
