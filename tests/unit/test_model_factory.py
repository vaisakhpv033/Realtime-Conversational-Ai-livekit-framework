"""Unit tests for Phase 3: Model & Turn-Detection Factory Abstractions.

Tests:
1. ModelFactory creation of GeminiLiveAdapter with default and custom hyperparameters.
2. Extended thinking configuration mapping (effort string -> ThinkingLevel enum, budget tokens, include_thoughts).
3. 50,000-token sliding window context compression presence.
4. Modality mapping ('audio', 'text' -> types.Modality enum).
5. VAD silence duration & disabled flag translation into RealtimeInputConfig (AutomaticActivityDetection).
6. ASR language locking and custom vocabulary biasing into AudioTranscriptionConfig.
7. Verification that instructions are NOT set on RealtimeModel (delegated to Agent).
8. Unsupported model provider error handling and descriptive diagnostics.
9. Custom adapter registration in ModelFactory.
10. TurnDetectionFactory TurnHandlingOptions mapping (interruption, resume_false, min words/duration, endpointing).
11. User-away timeout extraction and AgentSession compatibility.
"""

from __future__ import annotations

import os
from typing import Any, Optional
import pytest
from google.genai import types
from livekit.agents import AgentSession, TurnHandlingOptions, llm
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import google

from agent_framework.config.schema import (
    InstructionsConfig,
    ModelsConfig,
    SessionConfig,
    ThinkingConfig,
    TurnDetectionConfig,
    VADConfig,
    VoiceConfig,
)
from agent_framework.models.base import ModelAdapter, UnsupportedModelProviderError
from agent_framework.models.factory import ModelFactory
from agent_framework.models.gemini_live import GeminiLiveAdapter
from agent_framework.turn_detection.factory import TurnDetectionFactory


@pytest.fixture(autouse=True)
def ensure_google_api_key(monkeypatch):
    """Ensures a mock GOOGLE_API_KEY is present for RealtimeModel initialization."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-mock-google-api-key-12345")


@pytest.fixture(autouse=True)
def reset_factory_registry():
    """Ensures ModelFactory registry is clean before and after each test."""
    ModelFactory.reset_registry()
    yield
    ModelFactory.reset_registry()


def create_test_session_config(
    *,
    provider: str = "gemini",
    model: str = "gemini-3.8-live",
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 1024,
    modalities: Optional[list[str]] = None,
    thinking_enabled: bool = False,
    reasoning_effort: str = "medium",
    budget_tokens: Optional[int] = None,
    include_thoughts: bool = False,
    custom_vocabulary: Optional[list[str]] = None,
    voice_id: str = "Aoede",
    language: str = "en-US",
    vad_provider: str = "native",
    vad_silence_duration_ms: int = 1500,
    vad_disabled: bool = False,
    turn_detection_mode: str = "server_vad",
    allow_interruptions: bool = True,
    resume_false_interruption: bool = True,
    min_interruption_duration_ms: int = 500,
    min_interruption_words: int = 2,
    min_endpointing_delay_ms: int = 500,
    user_away_timeout_seconds: float = 20.0,
    system_prompt: str = "You are a customer service assistant.",
) -> SessionConfig:
    """Helper to construct a fully validated SessionConfig for unit testing."""
    return SessionConfig(
        models=ModelsConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            modalities=modalities or ["audio"],
            thinking=ThinkingConfig(
                enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
                budget_tokens=budget_tokens,
                include_thoughts=include_thoughts,
            ),
            custom_vocabulary=custom_vocabulary or [],
        ),
        instructions=InstructionsConfig(
            system_prompt=system_prompt,
        ),
        voice=VoiceConfig(
            provider=provider,
            voice_id=voice_id,
            language=language,
        ),
        vad=VADConfig(
            provider=vad_provider,
            silence_duration_ms=vad_silence_duration_ms,
            disabled=vad_disabled,
        ),
        turn_detection=TurnDetectionConfig(
            mode=turn_detection_mode,
            allow_interruptions=allow_interruptions,
            resume_false_interruption=resume_false_interruption,
            min_interruption_duration_ms=min_interruption_duration_ms,
            min_interruption_words=min_interruption_words,
            min_endpointing_delay_ms=min_endpointing_delay_ms,
            user_away_timeout_seconds=user_away_timeout_seconds,
        ),
    )


# ==============================================================================
# 1. Model Factory & Gemini Live Adapter Creation
# ==============================================================================
class TestModelFactoryAndGeminiLiveAdapter:
    """Tests for ModelFactory and GeminiLiveAdapter model creation."""

    def test_gemini_adapter_properties(self):
        """Verify provider name and model info inspection."""
        adapter = GeminiLiveAdapter()
        assert adapter.provider_name == "gemini"

        config = create_test_session_config(
            model="gemini-3.8-live",
            temperature=0.4,
            top_p=0.9,
            top_k=30,
            max_output_tokens=500,
            voice_id="Charon",
            custom_vocabulary=["term1", "term2"],
        )
        info = adapter.get_model_info(config)
        assert info["provider"] == "gemini"
        assert info["model"] == "gemini-3.8-live"
        assert info["effective_model"] == "gemini-3.8-live"
        assert info["temperature"] == 0.4
        assert info["top_p"] == 0.9
        assert info["top_k"] == 30
        assert info["max_output_tokens"] == 500
        assert info["voice"] == "Charon"
        assert info["custom_vocabulary_count"] == 2
        assert info["thinking_enabled"] is False

        # Verify effective_model reports promoted model when thinking enabled
        thinking_cfg = create_test_session_config(
            model="gemini-3.8-live",
            thinking_enabled=True,
        )
        thinking_info = adapter.get_model_info(thinking_cfg)
        assert thinking_info["effective_model"] == "gemini-3.8-live-extended-thinking"

    def test_factory_creates_realtime_model_default_params(self):
        """Verify ModelFactory creates RealtimeModel with default settings."""
        config = create_test_session_config()
        model = ModelFactory.create_model(config)

        assert isinstance(model, google.realtime.RealtimeModel)
        assert isinstance(model, llm.RealtimeModel)
        assert model._opts.model == "gemini-3.8-live"
        assert model._opts.voice == "Aoede"
        assert model._opts.temperature == 0.7
        assert model._opts.top_p == 0.95
        assert model._opts.top_k == 40
        assert model._opts.max_output_tokens == 1024

    def test_factory_creates_realtime_model_custom_params(self):
        """Verify custom hyperparameters and voice are properly wired."""
        config = create_test_session_config(
            model="gemini-3.1-flash-live-preview",
            voice_id="Puck",
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            max_output_tokens=512,
        )
        model = ModelFactory.create_model(config)

        assert model._opts.model == "gemini-3.1-flash-live-preview"
        assert model._opts.voice == "Puck"
        assert model._opts.temperature == 0.3
        assert model._opts.top_p == 0.8
        assert model._opts.top_k == 20
        assert model._opts.max_output_tokens == 512

    def test_factory_creates_realtime_model_with_extra_kwargs(self):
        """Verify ModelFactory forwards runtime kwargs (e.g. vertexai, project, location) without error."""
        config = create_test_session_config()
        model = ModelFactory.create_model(
            config,
            api_key="test-key",
            vertexai=True,
            project="test-gcp-project",
            location="us-central1",
        )
        assert model._opts.vertexai is True
        assert model._opts.project == "test-gcp-project"
        assert model._opts.location == "us-central1"

    def test_explicit_api_key_override(self):
        """Verify explicit api_key passed to kwargs overrides settings/env."""
        config = create_test_session_config()
        custom_key = "explicit-custom-api-key-999"
        model = ModelFactory.create_model(config, api_key=custom_key)

        assert model._opts.api_key == custom_key

    def test_provider_case_insensitivity(self):
        """Verify ModelFactory handles mixed-case provider strings."""
        config = create_test_session_config(provider="gemini")
        object.__setattr__(config.models, "provider", "GeMiNi")
        model = ModelFactory.create_model(config)
        assert model is not None
        assert model._opts.model == "gemini-3.8-live"

    def test_get_adapter_directly(self):
        """Verify ModelFactory.get_adapter returns configured GeminiLiveAdapter."""
        adapter = ModelFactory.get_adapter("gemini", api_key="test-key-direct")
        assert isinstance(adapter, GeminiLiveAdapter)
        assert adapter.provider_name == "gemini"
        assert adapter._api_key == "test-key-direct"


# ==============================================================================
# 2. Context Window Compression (50k sliding window)
# ==============================================================================
class TestContextWindowCompression:
    """Tests ensuring the 50,000-token sliding window context compression is always present."""

    def test_context_window_compression_configured(self):
        """Verify 50k sliding window compression enables long-running audio sessions."""
        config = create_test_session_config()
        model = ModelFactory.create_model(config)

        cwc = model._opts.context_window_compression
        assert cwc is not None
        assert cwc is not NOT_GIVEN
        assert isinstance(cwc, types.ContextWindowCompressionConfig)
        assert cwc.sliding_window is not None
        assert cwc.sliding_window.target_tokens == 50000


# ==============================================================================
# 3. Thinking Configuration Mapping
# ==============================================================================
class TestThinkingConfiguration:
    """Tests for extended thinking and reasoning translation."""

    @pytest.mark.parametrize(
        "effort_str, expected_level",
        [
            ("minimal", types.ThinkingLevel.LOW),
            ("low", types.ThinkingLevel.LOW),
            ("medium", types.ThinkingLevel.MEDIUM),
            ("high", types.ThinkingLevel.HIGH),
        ],
    )
    def test_thinking_level_translation(self, effort_str, expected_level):
        """Verify reasoning_effort string maps to correct ThinkingLevel enum and thinking_budget is None."""
        config = create_test_session_config(
            thinking_enabled=True,
            reasoning_effort=effort_str,
            budget_tokens=2048,
            include_thoughts=True,
        )
        model = ModelFactory.create_model(config)

        tc = model._opts.thinking_config
        assert tc is not None
        assert tc is not NOT_GIVEN
        assert isinstance(tc, types.ThinkingConfig)
        assert tc.thinking_level == expected_level
        assert tc.thinking_budget is None
        assert tc.include_thoughts is True

    def test_extended_thinking_model_auto_enabled(self):
        """Verify models with 'extended-thinking' in name configure thinking even if enabled=False."""
        config = create_test_session_config(
            model="gemini-3.8-live-extended-thinking",
            thinking_enabled=False,
            reasoning_effort="low",
            budget_tokens=1024,
        )
        model = ModelFactory.create_model(config)

        tc = model._opts.thinking_config
        assert tc is not None
        assert tc is not NOT_GIVEN
        assert isinstance(tc, types.ThinkingConfig)
        assert tc.thinking_level == types.ThinkingLevel.LOW
        assert tc.thinking_budget is None

    def test_gemini_38_thinking_budget_mutual_exclusivity(self):
        """Verify thinking_budget is strictly None when thinking_level is set, enforcing WebSocket protocol mutual exclusivity."""
        config = create_test_session_config(
            model="gemini-3.8-live-extended-thinking",
            thinking_enabled=True,
            reasoning_effort="high",
            budget_tokens=4096,
        )
        model = ModelFactory.create_model(config)
        tc = model._opts.thinking_config
        assert tc.thinking_level == types.ThinkingLevel.HIGH
        assert tc.thinking_budget is None

    def test_gemini_38_minimal_effort_normalized_to_low(self):
        """Verify reasoning_effort='minimal' is normalized to LOW for Gemini 3.8 extended thinking."""
        config = create_test_session_config(
            model="gemini-3.8-live-extended-thinking",
            thinking_enabled=True,
            reasoning_effort="minimal",
        )
        model = ModelFactory.create_model(config)
        tc = model._opts.thinking_config
        assert tc.thinking_level == types.ThinkingLevel.LOW
        assert tc.thinking_budget is None

    def test_standard_model_thinking_not_given(self):
        """Verify standard models without thinking enabled leave thinking_config as NOT_GIVEN."""
        config = create_test_session_config(
            model="gemini-3.8-live",
            thinking_enabled=False,
        )
        model = ModelFactory.create_model(config)

        assert model._opts.thinking_config is NOT_GIVEN

    @pytest.mark.parametrize(
        "model_input",
        [
            "gemini-3.8-live",
            "models/gemini-3.8-live",
            "google/gemini-3.8-live",
        ],
    )
    def test_model_promotion_with_google_and_models_prefix(self, model_input):
        """Verify prefixed gemini-3.8-live variants promote cleanly when thinking enabled."""
        config = create_test_session_config(
            model=model_input,
            thinking_enabled=True,
            reasoning_effort="high",
        )
        model = ModelFactory.create_model(config)

        assert model._opts.model == "gemini-3.8-live-extended-thinking"
        assert model._opts.thinking_config is not NOT_GIVEN
        assert model._opts.thinking_config.thinking_level == types.ThinkingLevel.HIGH

    @pytest.mark.parametrize(
        "model_input, expected_model",
        [
            ("google/gemini-3.8-live", "gemini-3.8-live"),
            ("models/gemini-3.8-live", "gemini-3.8-live"),
            ("google/gemini-3.8-live-extended-thinking", "gemini-3.8-live-extended-thinking"),
            ("models/gemini-3.8-live-extended-thinking", "gemini-3.8-live-extended-thinking"),
        ],
    )
    def test_prefixed_model_normalized_without_thinking(self, model_input, expected_model):
        """Verify google/ and models/ prefixes are stripped for clean API model endpoints."""
        config = create_test_session_config(
            model=model_input,
            thinking_enabled=False,
        )
        model = ModelFactory.create_model(config)
        assert model._opts.model == expected_model


# ==============================================================================
# 4. Modality Mapping
# ==============================================================================
class TestModalityMapping:
    """Tests for modality translation into google.genai.types.Modality."""

    def test_audio_only_modality(self):
        """Verify ['audio'] maps to [Modality.AUDIO]."""
        config = create_test_session_config(modalities=["audio"])
        model = ModelFactory.create_model(config)

        assert model._opts.response_modalities == [types.Modality.AUDIO]

    def test_multimodal_audio_and_text(self):
        """Verify ['audio', 'text'] maps to [Modality.AUDIO, Modality.TEXT]."""
        config = create_test_session_config(modalities=["audio", "text"])
        model = ModelFactory.create_model(config)

        assert model._opts.response_modalities == [types.Modality.AUDIO, types.Modality.TEXT]


# ==============================================================================
# 5. VAD and Realtime Input Configuration
# ==============================================================================
class TestVADConfiguration:
    """Tests for VAD silence duration and activity detection translation."""

    def test_vad_silence_duration_translation(self):
        """Verify VAD silence duration in ms is mapped to AutomaticActivityDetection."""
        config = create_test_session_config(vad_silence_duration_ms=1800, vad_disabled=False)
        model = ModelFactory.create_model(config)

        ric = model._opts.realtime_input_config
        assert ric is not None
        assert ric is not NOT_GIVEN
        assert isinstance(ric, types.RealtimeInputConfig)
        assert ric.automatic_activity_detection is not None
        assert ric.automatic_activity_detection.disabled is False
        assert ric.automatic_activity_detection.silence_duration_ms == 1800

    def test_vad_disabled_flag(self):
        """Verify vad_disabled=True sets AutomaticActivityDetection.disabled=True."""
        config = create_test_session_config(vad_silence_duration_ms=1000, vad_disabled=True)
        model = ModelFactory.create_model(config)

        ric = model._opts.realtime_input_config
        assert ric.automatic_activity_detection.disabled is True
        assert ric.automatic_activity_detection.silence_duration_ms == 1000

    def test_silero_vad_disables_server_activity_detection(self):
        """Verify silero VAD provider automatically disables Gemini server AutomaticActivityDetection."""
        config = create_test_session_config(
            vad_provider="silero",
            vad_silence_duration_ms=1200,
            vad_disabled=False,
        )
        model = ModelFactory.create_model(config)

        ric = model._opts.realtime_input_config
        assert ric.automatic_activity_detection.disabled is True


# ==============================================================================
# 6. ASR Language Locking & Custom Vocabulary Biasing
# ==============================================================================
class TestASRLanguageLockingAndVocabulary:
    """Tests for ASR language locking and custom speech recognition vocabulary."""

    def test_asr_language_locking_default(self):
        """Verify default language locks ASR language codes to en-US."""
        config = create_test_session_config(language="en-US")
        model = ModelFactory.create_model(config)

        assert model._opts.language == "en-US"
        assert model._opts.input_audio_transcription.language_codes == ["en-US"]
        assert model._opts.output_audio_transcription.language_codes == ["en-US"]

    def test_asr_language_locking_spanish(self):
        """Verify non-English language locks ASR language codes appropriately."""
        config = create_test_session_config(language="es-US")
        model = ModelFactory.create_model(config)

        assert model._opts.language == "es-US"
        assert model._opts.input_audio_transcription.language_codes == ["es-US"]
        assert model._opts.output_audio_transcription.language_codes == ["es-US"]

    def test_custom_vocabulary_biasing(self):
        """Verify caller custom_vocabulary is passed to input audio transcription."""
        vocab = ["root canal", "prophylaxis", "orthodontics", "Dr. Alvarez"]
        config = create_test_session_config(custom_vocabulary=vocab)
        model = ModelFactory.create_model(config)

        assert model._opts.input_audio_transcription.custom_vocabulary == vocab

    def test_empty_custom_vocabulary(self):
        """Verify empty custom vocabulary results in empty list."""
        config = create_test_session_config(custom_vocabulary=[])
        model = ModelFactory.create_model(config)

        assert model._opts.input_audio_transcription.custom_vocabulary == []

    def test_extra_languages_asr_locking(self):
        """Verify extra_languages passed in kwargs locks multilingual codes into transcription."""
        config = create_test_session_config(language="en-US")
        model = ModelFactory.create_model(config, extra_languages=["es-US", "hi-IN"])

        assert model._opts.input_audio_transcription.language_codes == ["en-US", "es-US", "hi-IN"]
        assert model._opts.output_audio_transcription.language_codes == ["en-US", "es-US", "hi-IN"]


# ==============================================================================
# 7. No Duplicate Instructions on RealtimeModel (commit ca0b6e7 adherence)
# ==============================================================================
class TestInstructionsOmission:
    """Verifies instructions are NOT set on RealtimeModel to avoid duplicate prompt token injection."""

    def test_instructions_not_set_on_realtime_model(self):
        """Confirm instructions remain NOT_GIVEN on RealtimeModel."""
        prompt = "You are a specialized diagnostic agent."
        config = create_test_session_config(system_prompt=prompt)
        model = ModelFactory.create_model(config)

        # Instructions MUST be NOT_GIVEN on RealtimeModel per architectural design
        assert model._opts.instructions is NOT_GIVEN

    def test_instructions_in_kwargs_ignored(self):
        """Confirm instructions passed maliciously/accidentally in kwargs are discarded."""
        config = create_test_session_config()
        model = ModelFactory.create_model(config, instructions="Injected instructions override")

        assert model._opts.instructions is NOT_GIVEN


# ==============================================================================
# 8. Error Handling & Registry Extensibility
# ==============================================================================
class TestModelFactoryRegistryAndErrors:
    """Tests for unsupported providers, error messages, and registry extensibility."""

    def test_unsupported_provider_raises_error(self):
        """Verify requesting an unsupported provider raises UnsupportedModelProviderError."""
        # Using a raw dict or patching provider in config
        config = create_test_session_config(provider="gemini")
        # Artificially set provider to unsupported
        object.__setattr__(config.models, "provider", "unsupported-llm")

        with pytest.raises(UnsupportedModelProviderError) as exc_info:
            ModelFactory.create_model(config)

        err_msg = str(exc_info.value)
        assert "Unsupported model provider 'unsupported-llm'" in err_msg
        assert "Registered providers are" in err_msg
        assert "gemini" in err_msg

    def test_custom_adapter_registration(self):
        """Verify registering a custom ModelAdapter allows dynamic creation."""

        class MockAnthropicAdapter(ModelAdapter):
            @property
            def provider_name(self) -> str:
                return "mock_anthropic"

            def build_realtime_model(self, session_config: SessionConfig, **kwargs: Any) -> llm.RealtimeModel:
                # Return a dummy RealtimeModel for testing
                return google.realtime.RealtimeModel(
                    model=session_config.models.model,
                    api_key="mock-key",
                )

        ModelFactory.register_adapter("mock_anthropic", MockAnthropicAdapter)

        config = create_test_session_config(provider="gemini", model="claude-3-opus")
        object.__setattr__(config.models, "provider", "mock_anthropic")

        created_model = ModelFactory.create_model(config)
        assert created_model is not None
        assert created_model._opts.model == "claude-3-opus"

    def test_valid_schema_provider_unregistered_raises_error(self):
        """Verify provider valid in schema ('openai') but unregistered in factory raises descriptive error."""
        config = create_test_session_config(provider="openai", model="gpt-4o-realtime")

        with pytest.raises(UnsupportedModelProviderError) as exc_info:
            ModelFactory.create_model(config)

        err_msg = str(exc_info.value)
        assert "Unsupported model provider 'openai'" in err_msg
        assert "gemini" in err_msg

    def test_model_factory_reset_and_unregister(self):
        """Verify unregister_adapter and reset_registry functions."""
        class DummyAdapter(ModelAdapter):
            @property
            def provider_name(self) -> str:
                return "dummy"

            def build_realtime_model(self, session_config: SessionConfig, **kwargs: Any) -> llm.RealtimeModel:
                return google.realtime.RealtimeModel(model="gemini-3.8-live")

        ModelFactory.register_adapter("dummy", DummyAdapter)
        assert "dummy" in ModelFactory._registry

        ModelFactory.unregister_adapter("dummy")
        assert "dummy" not in ModelFactory._registry

        ModelFactory.register_adapter("dummy2", DummyAdapter)
        ModelFactory.reset_registry()
        assert "dummy2" not in ModelFactory._registry
        assert "gemini" in ModelFactory._registry


# ==============================================================================
# 9. TurnDetectionFactory Tests
# ==============================================================================
class TestTurnDetectionFactory:
    """Tests for TurnDetectionFactory mapping to TurnHandlingOptions and inactivity timeouts."""

    def test_build_turn_handling_options_defaults(self):
        """Verify default turn detection settings map cleanly to TurnHandlingOptions."""
        config = create_test_session_config(
            allow_interruptions=True,
            resume_false_interruption=True,
            min_interruption_duration_ms=500,
            min_interruption_words=2,
            min_endpointing_delay_ms=500,
        )
        th = TurnDetectionFactory.build_turn_handling_options(config)

        assert isinstance(th, dict)
        assert "interruption" in th
        assert "endpointing" in th

        interruption = th["interruption"]
        assert interruption["enabled"] is True
        assert interruption["resume_false_interruption"] is True
        assert interruption["min_duration"] == 0.5  # 500ms converted to seconds
        assert interruption["min_words"] == 2

        endpointing = th["endpointing"]
        assert endpointing["min_delay"] == 0.5  # 500ms converted to seconds

    def test_build_turn_handling_options_custom(self):
        """Verify custom turn detection parameters map correctly with ms-to-seconds conversion."""
        config = create_test_session_config(
            allow_interruptions=False,
            resume_false_interruption=False,
            min_interruption_duration_ms=800,
            min_interruption_words=4,
            min_endpointing_delay_ms=1200,
        )
        th = TurnDetectionFactory.build_turn_handling_options(config)

        interruption = th["interruption"]
        assert interruption["enabled"] is False
        assert interruption["resume_false_interruption"] is False
        assert interruption["min_duration"] == 0.8  # 800ms -> 0.8s
        assert interruption["min_words"] == 4

        endpointing = th["endpointing"]
        assert endpointing["min_delay"] == 1.2  # 1200ms -> 1.2s

    def test_get_user_away_timeout(self):
        """Verify user away timeout extraction."""
        config = create_test_session_config(user_away_timeout_seconds=45.0)
        timeout = TurnDetectionFactory.get_user_away_timeout(config)
        assert timeout == 45.0
        assert isinstance(timeout, float)

    def test_turn_detection_nested_interruption_and_endpointing(self):
        """Verify TurnDetectionConfig accepts nested interruption and endpointing dicts."""
        td_dict = {
            "interruption": {
                "enabled": False,
                "resume_false_interruption": False,
                "min_duration": 0.6,
                "min_words": 3,
            },
            "endpointing": {
                "min_delay": 0.7,
            },
        }
        td = TurnDetectionConfig(**td_dict)
        assert td.allow_interruptions is False
        assert td.resume_false_interruption is False
        assert td.min_interruption_duration_ms == 600
        assert td.min_interruption_words == 3
        assert td.min_endpointing_delay_ms == 700

        config = create_test_session_config()
        object.__setattr__(config, "turn_detection", td)
        th = TurnDetectionFactory.build_turn_handling_options(config)
        assert th["interruption"]["enabled"] is False
        assert th["interruption"]["min_duration"] == 0.6
        assert th["endpointing"]["min_delay"] == 0.7

    def test_turn_detection_explicit_mode(self):
        """Verify explicit turn detection mode is reflected in TurnHandlingOptions."""
        config = create_test_session_config(turn_detection_mode="realtime_llm")
        th = TurnDetectionFactory.build_turn_handling_options(config)
        assert th.get("turn_detection") == "realtime_llm"

    @pytest.mark.asyncio
    async def test_turn_handling_options_compatible_with_agent_session(self):
        """Verify AgentSession accepts TurnHandlingOptions produced by TurnDetectionFactory."""
        config = create_test_session_config()
        th = TurnDetectionFactory.build_turn_handling_options(config)

        # AgentSession initialization with TurnHandlingOptions must not raise
        session = AgentSession(turn_handling=th)
        assert session is not None
