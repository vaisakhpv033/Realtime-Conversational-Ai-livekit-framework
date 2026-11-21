"""Abstract base classes and exceptions for model adapter implementations.

Provides standard abstractions for building LiveKit RealtimeModel instances
across multiple LLM providers (Google Gemini Live, OpenAI Realtime, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from livekit.agents import llm

from agent_framework.config.schema import SessionConfig


class UnsupportedModelProviderError(ValueError):
    """Raised when an unsupported or unregistered model provider is requested."""

    pass


class ModelAdapter(ABC):
    """Abstract adapter defining the contract for model providers in the agent framework."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical provider name (e.g. 'gemini', 'openai')."""
        pass

    @abstractmethod
    def build_realtime_model(
        self, session_config: SessionConfig, **kwargs: Any
    ) -> llm.RealtimeModel:
        """Builds and returns a configured LiveKit RealtimeModel instance.

        Args:
            session_config: Validated session configuration containing model hyperparameters,
                voice, ASR languages, and VAD thresholds.
            **kwargs: Additional provider-specific runtime options (e.g. api_key overrides).

        Returns:
            A configured llm.RealtimeModel instance ready for AgentSession.
        """
        pass

    def get_model_info(self, session_config: SessionConfig) -> Dict[str, Any]:
        """Returns inspection metadata describing the configured model.

        Args:
            session_config: The session configuration to inspect.

        Returns:
            Dictionary containing model identity, hyperparameters, and capabilities.
        """
        return {
            "provider": self.provider_name,
            "model": session_config.models.model,
            "temperature": session_config.models.temperature,
            "top_p": session_config.models.top_p,
            "top_k": session_config.models.top_k,
            "max_output_tokens": session_config.models.max_output_tokens,
            "modalities": session_config.models.modalities,
            "voice": session_config.voice.voice_id,
            "thinking_enabled": session_config.models.thinking.enabled,
            "reasoning_effort": session_config.models.thinking.reasoning_effort,
            "custom_vocabulary_count": len(session_config.models.custom_vocabulary),
        }
