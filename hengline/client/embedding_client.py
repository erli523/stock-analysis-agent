"""
Embedding client factory.
"""

from typing import List, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import Field

from config.config import get_embedding_config, mask_sensitive_config
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class CompatibleOpenAIEmbedding(BaseEmbedding):
    """Embedding wrapper for OpenAI-compatible providers with custom model names."""

    model_name: str = Field(...)
    api_key: str = Field(...)
    base_url: str = Field(...)
    timeout: float = Field(default=60.0)

    def _embed(self, text: str) -> List[float]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        response = client.embeddings.create(model=self.model_name, input=text)
        return list(response.data[0].embedding)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)


def _requires_compatible_wrapper(model_name: str, base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    normalized_base = base_url.rstrip("/").lower()
    if "api.openai.com" in normalized_base:
        return False
    openai_known_prefixes = ("text-embedding-ada-", "text-embedding-3-")
    return not model_name.startswith(openai_known_prefixes) or "compatible-mode" in normalized_base


def get_embedding_client(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseEmbedding:
    """Create an embedding model instance from config."""
    try:
        embedding_config = get_embedding_config()
        debug(f"Embedding config: {mask_sensitive_config(embedding_config)}")

        if model_type is None:
            model_type = embedding_config.get("provider", "openai")
        model_type = str(model_type).lower()

        if model_name is None:
            model_name = embedding_config.get("model_name", embedding_config.get("model", "text-embedding-3-small"))

        if not isinstance(kwargs, dict):
            kwargs = {}

        debug(f"Creating embedding model: type={model_type}, name={model_name}")

        if model_type == "openai":
            base_url = embedding_config.get("base_url")
            api_key = embedding_config.get("api_key")
            timeout = float(embedding_config.get("timeout", 60))

            if _requires_compatible_wrapper(model_name, base_url):
                return CompatibleOpenAIEmbedding(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )

            config_kwargs = {
                "api_base": base_url,
                "api_key": api_key,
                "timeout": timeout,
            }
            merged_kwargs = {**config_kwargs, **kwargs}
            merged_kwargs = {key: value for key, value in merged_kwargs.items() if value is not None}
            debug(f"OpenAI embedding kwargs: {mask_sensitive_config(merged_kwargs)}")
            return OpenAIEmbedding(model=model_name, **merged_kwargs)

        if model_type == "huggingface":
            config_kwargs = {
                "model_name": model_name,
                "token": embedding_config.get("token"),
                "cache_folder": embedding_config.get("cache_folder", "data/embeddings"),
                "device": embedding_config.get("device", "cpu"),
            }
            merged_kwargs = {**config_kwargs, **kwargs}
            merged_kwargs = {key: value for key, value in merged_kwargs.items() if value is not None}
            debug(f"HuggingFace embedding kwargs: {mask_sensitive_config(merged_kwargs)}")
            return HuggingFaceEmbedding(**merged_kwargs)

        if model_type == "ollama":
            config_kwargs = {
                "base_url": embedding_config.get("base_url", "http://localhost:11434"),
                "request_timeout": embedding_config.get("timeout"),
            }
            merged_kwargs = {**config_kwargs, **kwargs}
            merged_kwargs = {key: value for key, value in merged_kwargs.items() if value is not None}
            debug(f"Ollama embedding kwargs: {mask_sensitive_config(merged_kwargs)}")
            return OllamaEmbedding(model_name=model_name, **merged_kwargs)

        raise ValueError(f"Unsupported embedding model type: {model_type}")

    except Exception as exc:
        print_log_exception()
        error(f"Failed to create embedding model: {exc}; falling back to OpenAI default")
        try:
            return OpenAIEmbedding(model="text-embedding-3-small")
        except Exception as default_error:
            error(f"Default embedding model also failed: {default_error}")
            raise RuntimeError("Unable to initialize any embedding model") from default_error
