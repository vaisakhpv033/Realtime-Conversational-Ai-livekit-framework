"""Model factory dynamically instantiating provider adapters from session metadata.

Maintains an extensible registry of ModelAdapter implementations and builds
configured RealtimeModel instances for LiveKit sessions.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Type

from livekit.agents import llm

from agent_framework.config.schema import SessionConfig
from agent_framework.models.base import ModelAdapter, UnsupportedModelProviderError
from agent_framework.models.gemini_live import GeminiLiveAdapter

logger = logging.getLogger("agent_framework.models.factory")


class ModelFactory:
    """Factory creating provider-specific RealtimeModel instances from SessionConfig."""

    _registry: ClassVar[Dict[str, Type[ModelAdapter]]] = {}

    @classmethod
    def register_adapter(cls, provider: str, adapter_cls: Type[ModelAdapter]) -> None:
        """Registers a model adapter class for a given provider key.

        Args:
            provider: Provider name (case-insensitive, e.g. 'gemini', 'openai').
            adapter_cls: Subclass of ModelAdapter implementing build_realtime_model.
        """
        key = provider.strip().lower()
        cls._registry[key] = adapter_cls
        logger.debug(f"Registered model adapter for provider '{key}': {adapter_cls.__name__}")

    @classmethod
    def unregister_adapter(cls, provider: str) -> None:
        """Unregisters a model adapter class if present.

        Args:
            provider: Provider name (case-insensitive).
        """
        key = provider.strip().lower()
        cls._registry.pop(key, None)

    @classmethod
    def reset_registry(cls) -> None:
        """Resets the adapter registry to default built-in providers."""
        cls._registry.clear()
        cls.register_adapter("gemini", GeminiLiveAdapter)

    @classmethod
    def get_adapter(cls, provider: str, **kwargs: Any) -> ModelAdapter:
        """Instantiates and returns the ModelAdapter registered for the provider.

        Args:
            provider: Canonical provider name (e.g. 'gemini').
            **kwargs: Arguments forwarded to the adapter constructor.

        Returns:
            An instantiated ModelAdapter.

        Raises:
            UnsupportedModelProviderError: If the provider is not registered.
        """
        key = provider.strip().lower()
        adapter_cls = cls._registry.get(key)
        if adapter_cls is None:
            available = sorted(cls._registry.keys())
            raise UnsupportedModelProviderError(
                f"Unsupported model provider '{provider}'. "
                f"Registered providers are: {available}"
            )
        return adapter_cls(**kwargs)

    @classmethod
    def create_model(
        cls, session_config: SessionConfig, **kwargs: Any
    ) -> llm.RealtimeModel:
        """Creates a fully configured RealtimeModel from validated SessionConfig.

        Args:
            session_config: Validated session configuration containing models and voice sections.
            **kwargs: Additional runtime options forwarded to the adapter (e.g. api_key).

        Returns:
            A configured llm.RealtimeModel instance.

        Raises:
            UnsupportedModelProviderError: If session_config.models.provider is not registered.
        """
        provider = session_config.models.provider
        adapter = cls.get_adapter(provider, **kwargs)
        return adapter.build_realtime_model(session_config, **kwargs)


# Register default built-in adapters
ModelFactory.reset_registry()
