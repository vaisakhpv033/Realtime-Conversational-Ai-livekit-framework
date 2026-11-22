"""Google Gemini 3.8 / 3.1 Live model adapter implementation.

Implements the ModelAdapter interface for Google Gemini Live Speech-to-Speech (STS)
streaming via `livekit.plugins.google.realtime.RealtimeModel`.

Key Architectural Features:
1. 50,000-token sliding window context compression for unlimited session duration.
2. Extended thinking and reasoning support with ThinkingLevel and budget configuration.
3. Locked ASR language codes to prevent multilingual transcription drift.
4. Dynamic VAD silence sensitivity mapped to AutomaticActivityDetection.
5. Strict omission of instructions on RealtimeModel (delegated to Agent) to avoid prompt duplication.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from google.genai import types
from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import google

from agent_framework.config.schema import SessionConfig
from agent_framework.config.settings import settings
from agent_framework.models.base import ModelAdapter

logger = logging.getLogger("agent_framework.models.gemini_live")

# Standard modality mapping
_MODALITY_MAP = {
    "audio": types.Modality.AUDIO,
    "text": types.Modality.TEXT,
}

# Thinking level mapping
_THINKING_LEVEL_MAP = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


class GeminiLiveAdapter(ModelAdapter):
    """Adapter for Google Gemini Live Realtime models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initializes the adapter with optional explicit API key and model options.

        Args:
            api_key: Optional Google API key override. If not supplied, falls back to
                settings.google_api_key or GOOGLE_API_KEY environment variable.
            **kwargs: Additional provider-specific default options (e.g. vertexai, project).
        """
        self._api_key = api_key
        self._default_kwargs = kwargs

    @property
    def provider_name(self) -> str:
        """Returns the canonical provider name 'gemini'."""
        return "gemini"

    def get_model_info(self, session_config: SessionConfig) -> dict[str, Any]:
        """Returns inspection metadata describing the configured model, including effective model."""
        info = super().get_model_info(session_config)
        clean_model = session_config.models.model.lower().removeprefix("models/").removeprefix("google/")
        if clean_model == "gemini-3.8-live" and session_config.models.thinking.enabled:
            info["effective_model"] = "gemini-3.8-live-extended-thinking"
        else:
            info["effective_model"] = session_config.models.model
        return info

    def build_realtime_model(
        self, session_config: SessionConfig, **kwargs: Any
    ) -> llm.RealtimeModel:
        """Builds a configured Gemini Live RealtimeModel from session metadata.

        Args:
            session_config: Validated SessionConfig containing model, voice, ASR,
                and VAD parameters.
            **kwargs: Additional runtime overrides (e.g. api_key, vertexai, project).

        Returns:
            google.realtime.RealtimeModel instance.
        """
        # Merge adapter defaults with call-site kwargs
        merged_kwargs = {**self._default_kwargs, **kwargs}

        # Strict omission of instructions per commit ca0b6e7
        merged_kwargs.pop("instructions", None)

        model_name = session_config.models.model
        voice_id = session_config.voice.voice_id
        temperature = session_config.models.temperature

        # 0. Resilient model resolution for thinking models
        # On Gemini API, gemini-3.8-live disallows thinking_level; promote to extended-thinking variant
        clean_model = model_name.removeprefix("models/").removeprefix("google/")
        if (
            clean_model.lower() == "gemini-3.8-live"
            and session_config.models.thinking.enabled
        ):
            model_name = "gemini-3.8-live-extended-thinking"
            logger.info("Promoted 'gemini-3.8-live' to 'gemini-3.8-live-extended-thinking' due to thinking.enabled=True")
        else:
            model_name = clean_model

        # 1. Resolve API key
        api_key = (
            merged_kwargs.pop("api_key", None)
            or self._api_key
            or settings.google_api_key
            or os.environ.get("GOOGLE_API_KEY")
            or NOT_GIVEN
        )

        # 2. Map modalities (e.g. ['audio'] -> [types.Modality.AUDIO])
        modalities: list[types.Modality] = []
        for mod_str in session_config.models.modalities:
            clean_mod = mod_str.strip().lower()
            if clean_mod in _MODALITY_MAP:
                modalities.append(_MODALITY_MAP[clean_mod])
        if not modalities:
            modalities = [types.Modality.AUDIO]

        # 3. Resolve BCP-47 language & locked ASR language codes
        bcp47_language = settings.resolve_bcp47_language(session_config.voice.language)
        extra_languages = merged_kwargs.pop("extra_languages", None)
        asr_language_codes = settings.resolve_asr_language_codes(
            language_code=session_config.voice.language,
            extra_languages=extra_languages,
        )

        # 4. Context Window Compression: 50,000-token sliding window
        # Audio tokens accumulate at ~25 tokens/sec. Without compression,
        # audio-only sessions are limited to ~15 minutes.
        context_window_compression = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(
                target_tokens=50000,
            ),
        )

        # 5. Resolve Extended Thinking configuration
        # Enabled if explicitly requested or if model is an extended-thinking variant
        is_extended_thinking = "extended-thinking" in model_name.lower()
        if session_config.models.thinking.enabled or is_extended_thinking:
            effort_str = (session_config.models.thinking.reasoning_effort or "medium").strip().lower()
            is_gemini_38 = "3.8" in model_name.lower() or is_extended_thinking
            # Gemini 3.8 Live Extended Thinking does not support MINIMAL; map to LOW
            if is_gemini_38 and effort_str == "minimal":
                thinking_level = types.ThinkingLevel.LOW
            else:
                thinking_level = _THINKING_LEVEL_MAP.get(effort_str, types.ThinkingLevel.MEDIUM)

            # Gemini Live WebSocket protocol strictly rejects setting both thinking_budget and thinking_level
            # (WebSocket close code 1007). In Gemini 3.8, thinking_budget is deprecated and thinking_level
            # is authoritative. Enforce strict mutual exclusivity: thinking_budget must be None when thinking_level is set.
            thinking_budget: Optional[int]
            if is_gemini_38 or thinking_level is not None:
                thinking_budget = None
                if session_config.models.thinking.budget_tokens is not None:
                    logger.warning(
                        f"budget_tokens={session_config.models.thinking.budget_tokens} ignored for {model_name}; "
                        f"thinking_level={thinking_level} is authoritative and mutually exclusive on Gemini Live."
                    )
            else:
                thinking_budget = session_config.models.thinking.budget_tokens

            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                include_thoughts=session_config.models.thinking.include_thoughts,
            )
            logger.debug(
                f"Configured ThinkingConfig for {model_name}: "
                f"level={thinking_level}, budget={thinking_budget}, "
                f"include_thoughts={session_config.models.thinking.include_thoughts}"
            )
        else:
            thinking_config = NOT_GIVEN

        # 6. ASR Audio Transcription with Language Locking & Custom Vocabulary Biasing
        input_audio_transcription = types.AudioTranscriptionConfig(
            language_codes=asr_language_codes,
            custom_vocabulary=session_config.models.custom_vocabulary if session_config.models.custom_vocabulary else [],
        )
        output_audio_transcription = types.AudioTranscriptionConfig(
            language_codes=asr_language_codes,
        )

        # 7. VAD Automatic Activity Detection
        vad_disabled = session_config.vad.disabled or (session_config.vad.provider == "silero")
        silence_duration_ms = session_config.vad.silence_duration_ms
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=vad_disabled,
                silence_duration_ms=silence_duration_ms,
            )
        )

        # 8. Build and return RealtimeModel
        # NOTE: System instructions are deliberately NOT set here (omitted / NOT_GIVEN)
        # to prevent duplicate prompt injection into Gemini Live (instructions are set
        # exclusively on livekit.agents.Agent per commit ca0b6e7).
        return google.realtime.RealtimeModel(
            model=model_name,
            voice=voice_id,
            temperature=temperature,
            top_p=session_config.models.top_p,
            top_k=session_config.models.top_k,
            max_output_tokens=session_config.models.max_output_tokens,
            api_key=api_key,
            modalities=modalities,
            language=bcp47_language,
            thinking_config=thinking_config,
            input_audio_transcription=input_audio_transcription,
            output_audio_transcription=output_audio_transcription,
            realtime_input_config=realtime_input_config,
            context_window_compression=context_window_compression,
            **merged_kwargs,
        )
