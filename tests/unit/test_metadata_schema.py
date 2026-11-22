"""Comprehensive test suite for Phase 1: Metadata Schema & Strict Validation Engine.

Tests:
1. Valid payloads (complete 10-section payload and minimal valid payload).
2. Schema defaults and cross-field post-validation.
3. Fail-fast validation errors (missing required keys, invalid ranges, extra keys).
4. Secret substitution ({{env:KEY}} resolution, strict fail-fast on missing keys).
5. Prompt templating ({key} / {{key}} interpolation, spoken dialogue rules).
6. Cascading overlays (Job -> Room -> Participant priority, targeting by identity).
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock
from livekit.agents import JobContext

from agent_framework.config.schema import (
    AgentConfig,
    AgentMetadataContract,
    FirstTurnConfig,
    InstructionsConfig,
    ModelsConfig,
    ReportingConfig,
    SessionConfig,
    SessionLifecycleConfig,
    ThinkingConfig,
    ToolConfig,
    TurnDetectionConfig,
    VADConfig,
    VoiceConfig,
)
from agent_framework.config.validator import (
    MetadataValidationError,
    validate_metadata,
)
from agent_framework.config.resolver import (
    deep_merge,
    extract_participant_metadata,
    extract_session_metadata,
    parse_layer_metadata,
    resolve_cascading_metadata,
)
from agent_framework.config.settings import (
    LANGUAGE_BCP47_MAP,
    FrameworkSettings,
    settings,
)
from agent_framework.utils.secrets import (
    SecretResolutionError,
    resolve_secret,
    resolve_secrets,
)
from agent_framework.utils.templating import (
    TemplateInterpolationError,
    interpolate_prompt,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def minimal_valid_payload() -> dict:
    """Minimal payload containing only the strictly required sections."""
    return {
        "models": {
            "provider": "gemini",
            "model": "gemini-3.8-live",
        },
        "instructions": {
            "system_prompt": "You are a helpful customer support agent.",
        },
        "voice": {
            "voice_id": "Aoede",
        },
    }


@pytest.fixture
def full_valid_payload() -> dict:
    """Complete 10-section payload adhering strictly to Section 2 specifications."""
    return {
        "version": "1.0",
        "agent": {
            "name": "support-voice-bot",
            "role": "customer_support",
            "description": "Tier 1 customer support voice assistant",
        },
        "models": {
            "provider": "gemini",
            "model": "gemini-3.8-live-extended-thinking",
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 30,
            "max_output_tokens": 2048,
            "modalities": ["audio", "text"],
            "thinking": {
                "enabled": True,
                "reasoning_effort": "high",
                "budget_tokens": 2048,
                "include_thoughts": True,
            },
            "custom_vocabulary": ["Acme Corp", "Widget Pro", "SKU-9982"],
        },
        "instructions": {
            "system_prompt": "Hello {user_name}, you are an assistant for {company_name}.",
            "first_turn": {
                "speaker": "agent",
                "mode": "say",
                "text": "Hello, thank you for calling support. How can I help you today?",
            },
            "context_data": {
                "user_name": "Alice",
                "company_name": "Acme Corp",
            },
            "spoken_dialogue_rules": True,
        },
        "voice": {
            "provider": "gemini",
            "voice_id": "Puck",
            "language": "en-US",
            "speed": 1.1,
        },
        "vad": {
            "provider": "native",
            "silence_duration_ms": 1200,
            "prefix_padding_ms": 250,
            "threshold": 0.6,
        },
        "turn_detection": {
            "mode": "server_vad",
            "allow_interruptions": True,
            "min_interruption_duration_ms": 400,
            "min_interruption_words": 1,
            "resume_false_interruption": True,
            "min_endpointing_delay_ms": 600,
            "user_away_timeout_seconds": 30.0,
        },
        "session": {
            "max_duration_seconds": 2400,
            "inactivity_timeout_seconds": 180,
            "participant_join_timeout_seconds": 45.0,
            "delete_room_on_terminate": True,
            "target_participant_identity": "customer_user_99",
            "recording": {
                "enabled": True,
                "audio": True,
                "traces": True,
            },
        },
        "tools": [
            {
                "name": "lookup_order",
                "type": "webhook",
                "description": "Looks up an order status by order ID",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                "config": {
                    "url": "https://api.acme.com/v1/orders",
                    "method": "POST",
                    "headers": {"Authorization": "Bearer {{env:ACME_API_KEY}}"},
                    "timeout_seconds": 5.0,
                },
            },
            {
                "name": "end_call",
                "type": "builtin",
                "description": "Gracefully ends call",
                "config": {
                    "delete_room": True,
                    "farewell_instruction": "Say a brief goodbye before disconnecting",
                },
            },
        ],
        "reporting": {
            "enabled": True,
            "endpoint": {
                "url": "https://telemetry.acme.com/v1/calls",
                "method": "POST",
                "headers": {"X-API-Key": "{{env:REPORTING_KEY}}"},
                "timeout_seconds": 15.0,
            },
            "auth": {
                "type": "bearer",
                "secret_ref": "env:REPORTING_SECRET",
            },
            "retry_policy": {
                "max_retries": 3,
                "initial_delay_ms": 1500,
                "backoff_factor": 2.5,
            },
            "security": {
                "validate_ssrf": True,
                "allow_local": False,
            },
            "payload_config": {
                "include_chat_history": True,
                "include_model_usage": True,
                "include_events": True,
                "custom_fields": {"department": "billing"},
            },
        },
        "metadata": {
            "account_id": "ACC-54321",
            "tenant_id": "tenant-001",
        },
    }


# ==============================================================================
# 1. Valid Payload Tests
# ==============================================================================
class TestValidPayloads:
    """Verifies that correctly structured payloads parse into SessionConfig cleanly."""

    def test_minimal_valid_payload_parsing(self, minimal_valid_payload):
        config = validate_metadata(minimal_valid_payload)
        assert isinstance(config, SessionConfig)
        assert config.models.provider == "gemini"
        assert config.models.model == "gemini-3.8-live"
        assert config.voice.voice_id == "Aoede"
        # Cross-field default: voice.provider defaults to models.provider
        assert config.voice.provider == "gemini"
        # Default VAD and Session settings
        assert config.vad.silence_duration_ms == 1500
        assert config.session.max_duration_seconds == 1800
        assert config.session.delete_room_on_terminate is True
        assert config.reporting.enabled is False
        assert config.metadata == {}

    def test_full_valid_payload_parsing(self, full_valid_payload):
        # We test validation without resolving secrets first
        config = validate_metadata(full_valid_payload)
        assert config.version == "1.0"
        assert config.agent.name == "support-voice-bot"
        assert config.agent.role == "customer_support"
        assert config.models.temperature == 0.5
        assert config.models.top_k == 30
        assert config.models.thinking.enabled is True
        assert config.models.thinking.reasoning_effort == "high"
        assert config.instructions.first_turn.speaker == "agent"
        assert config.instructions.first_turn.mode == "say"
        assert len(config.tools) == 2
        assert config.tools[0].name == "lookup_order"
        assert config.tools[0].type == "webhook"
        assert config.tools[1].name == "end_call"
        assert config.tools[1].type == "builtin"
        assert config.reporting.enabled is True
        assert config.reporting.endpoint.url == "https://telemetry.acme.com/v1/calls"
        assert config.metadata["account_id"] == "ACC-54321"

    def test_json_string_input_parsing(self, minimal_valid_payload):
        json_str = json.dumps(minimal_valid_payload)
        config = validate_metadata(json_str)
        assert isinstance(config, SessionConfig)
        assert config.models.provider == "gemini"

    def test_reporting_auto_enables_when_endpoint_url_provided(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["reporting"] = {
            "endpoint": {
                "url": "https://api.example.com/webhook",
            }
        }
        config = validate_metadata(payload)
        # Should auto-enable reporting when endpoint url is supplied without explicit enabled=False
        assert config.reporting.enabled is True
        assert config.reporting.endpoint.url == "https://api.example.com/webhook"

    def test_session_config_idempotent_validation(self, minimal_valid_payload):
        config1 = validate_metadata(minimal_valid_payload)
        config2 = validate_metadata(config1)
        assert config1 is config2


# ==============================================================================
# 2. Strict Fail-Fast Validation Error Tests
# ==============================================================================
class TestValidationFailures:
    """Verifies that missing keys, invalid types, or out-of-bounds values fail fast."""

    def test_missing_required_top_level_sections(self):
        # Missing 'models'
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata({
                "instructions": {"system_prompt": "Hello"},
                "voice": {"voice_id": "Aoede"},
            })
        assert "models" in str(exc.value)

        # Missing 'instructions'
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata({
                "models": {"provider": "gemini", "model": "gemini-3.8-live"},
                "voice": {"voice_id": "Aoede"},
            })
        assert "instructions" in str(exc.value)

        # Missing 'voice'
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata({
                "models": {"provider": "gemini", "model": "gemini-3.8-live"},
                "instructions": {"system_prompt": "Hello"},
            })
        assert "voice" in str(exc.value)

    def test_empty_input_payloads(self):
        with pytest.raises(MetadataValidationError):
            validate_metadata("")
        with pytest.raises(MetadataValidationError):
            validate_metadata("   ")
        with pytest.raises(MetadataValidationError):
            validate_metadata("{invalid_json:")
        with pytest.raises(MetadataValidationError):
            validate_metadata(12345)
        with pytest.raises(MetadataValidationError):
            validate_metadata(["not", "a", "dict"])

    def test_extra_top_level_forbidden_keys(self, minimal_valid_payload):
        # Greenfield rejects legacy unmapped keys at top level (e.g. legacy trainee_id)
        payload = dict(minimal_valid_payload)
        payload["trainee_id"] = "citrus-rep-01"
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "trainee_id" in str(exc.value)

    def test_invalid_model_provider(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["models"] = {
            "provider": "unsupported-llm",
            "model": "foo-bar",
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "provider" in str(exc.value)

    def test_temperature_out_of_range(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["models"]["temperature"] = 3.5  # Max is 2.0
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "temperature" in str(exc.value)

        payload["models"]["temperature"] = -0.1
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "temperature" in str(exc.value)

    def test_empty_system_prompt(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {"system_prompt": "   "}
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "system_prompt" in str(exc.value)

    def test_first_turn_say_without_text(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "Helpful agent",
            "first_turn": {"mode": "say"},  # Missing text
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "first_turn.text" in str(exc.value)

    def test_vad_silence_duration_out_of_range(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["vad"] = {"silence_duration_ms": 100}  # Min is 200
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "silence_duration_ms" in str(exc.value)

        payload["vad"] = {"silence_duration_ms": 5000}  # Max is 3000
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "silence_duration_ms" in str(exc.value)

    def test_webhook_tool_missing_url(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "calendar_lookup",
            "type": "webhook",
            "config": {},  # missing 'url'
        }]
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "url" in str(exc.value)

    def test_reporting_enabled_without_url(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["reporting"] = {
            "enabled": True,
            "endpoint": None,
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "reporting.endpoint.url" in str(exc.value)


# ==============================================================================
# 3. Secret Resolution Tests
# ==============================================================================
class TestSecretResolution:
    """Verifies {{env:KEY}} and env:KEY secret substitution."""

    def test_single_mustache_secret_resolution(self):
        mock_env = {"MY_API_KEY": "super-secret-token-123"}
        result = resolve_secret("Bearer {{env:MY_API_KEY}}", env=mock_env)
        assert result == "Bearer super-secret-token-123"

    def test_multiple_secrets_in_one_string(self):
        mock_env = {"HOST": "api.acme.com", "TOKEN": "xyz999"}
        template = "https://{{env:HOST}}/endpoint?key={{env:TOKEN}}"
        result = resolve_secret(template, env=mock_env)
        assert result == "https://api.acme.com/endpoint?key=xyz999"

    def test_prefix_syntax_resolution(self):
        mock_env = {"REPORTING_KEY": "rep-abc-456"}
        result = resolve_secret("env:REPORTING_KEY", env=mock_env)
        assert result == "rep-abc-456"

    def test_strict_mode_raises_on_missing_secret(self):
        mock_env = {"OTHER_KEY": "123"}
        with pytest.raises(SecretResolutionError) as exc:
            resolve_secret("Bearer {{env:MISSING_KEY}}", env=mock_env, strict=True)
        assert "MISSING_KEY" in str(exc.value)

    def test_non_strict_mode_preserves_placeholder(self):
        mock_env = {}
        result = resolve_secret("Bearer {{env:MISSING_KEY}}", env=mock_env, strict=False)
        assert result == "Bearer {{env:MISSING_KEY}}"

    def test_recursive_nested_dict_and_list_resolution(self):
        mock_env = {
            "SECRET_URL": "https://auth.acme.com",
            "SECRET_TOKEN": "token-xyz",
        }
        data = {
            "reporting": {
                "url": "{{env:SECRET_URL}}",
                "headers": ["Authorization: Bearer {{env:SECRET_TOKEN}}"],
                "raw_number": 42,
            }
        }
        resolved = resolve_secrets(data, env=mock_env, strict=True)
        assert resolved["reporting"]["url"] == "https://auth.acme.com"
        assert resolved["reporting"]["headers"] == ["Authorization: Bearer token-xyz"]
        assert resolved["reporting"]["raw_number"] == 42

    def test_validate_metadata_with_secret_resolution(self, full_valid_payload):
        mock_env = {
            "ACME_API_KEY": "live_acme_secret_999",
            "REPORTING_KEY": "live_reporting_key_888",
            "REPORTING_SECRET": "live_auth_secret_777",
        }
        config = validate_metadata(full_valid_payload, resolve_env_secrets=True, env=mock_env)
        assert config.tools[0].config["headers"]["Authorization"] == "Bearer live_acme_secret_999"
        assert config.reporting.endpoint.headers["X-API-Key"] == "live_reporting_key_888"
        assert config.reporting.auth.secret_ref == "live_auth_secret_777"


# ==============================================================================
# 4. Prompt Templating & Dialogue Rules Tests
# ==============================================================================
class TestPromptTemplating:
    """Verifies {key} and {{key}} interpolation into system prompts."""

    def test_single_brace_interpolation(self):
        prompt = "Hello {user_name}, welcome to {service_name}."
        ctx = {"user_name": "Marcus", "service_name": "ClinicOps"}
        interpolated = interpolate_prompt(prompt, ctx)
        assert interpolated == "Hello Marcus, welcome to ClinicOps."

    def test_double_brace_interpolation(self):
        prompt = "Hello {{user_name}}, welcome to {{service_name}}."
        ctx = {"user_name": "Marcus", "service_name": "ClinicOps"}
        interpolated = interpolate_prompt(prompt, ctx)
        assert interpolated == "Hello Marcus, welcome to ClinicOps."

    def test_json_syntax_preserved_when_not_in_context(self):
        # A prompt with literal JSON instructions should not crash or be corrupted
        prompt = 'Format response as: {"status": "ok", "message": "{custom_message}"}'
        ctx = {"custom_message": "Session completed"}
        interpolated = interpolate_prompt(prompt, ctx)
        assert interpolated == 'Format response as: {"status": "ok", "message": "Session completed"}'

    def test_strict_mode_raises_on_missing_context_var(self):
        prompt = "Welcome {user_name}, your balance is {balance}."
        ctx = {"user_name": "Alice"}
        with pytest.raises(TemplateInterpolationError) as exc:
            interpolate_prompt(prompt, ctx, strict=True)
        assert "balance" in str(exc.value)

    def test_effective_prompt_appends_spoken_dialogue_rules(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "You are Marcus, caller ID {caller_id}.",
            "context_data": {"caller_id": "9921"},
            "spoken_dialogue_rules": True,
        }
        config = validate_metadata(payload)
        effective = config.get_effective_system_prompt()
        assert "You are Marcus, caller ID 9921." in effective
        assert "Conversational speech guidelines:" in effective
        assert "typically 1-2 sentences per turn" in effective

    def test_effective_prompt_omits_spoken_rules_when_disabled(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "You are Marcus, caller ID {caller_id}.",
            "context_data": {"caller_id": "9921"},
            "spoken_dialogue_rules": False,
        }
        config = validate_metadata(payload)
        effective = config.get_effective_system_prompt()
        assert effective == "You are Marcus, caller ID 9921."
        assert "Conversational speech guidelines:" not in effective


# ==============================================================================
# 5. Cascading Overlays & Participant Targeting Tests
# ==============================================================================
class TestCascadingResolver:
    """Verifies multi-source metadata resolution: Job -> Room -> Participant."""

    def test_deep_merge_basic(self):
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        overlay = {"b": 2, "nested": {"y": 99, "z": 30}}
        merged = deep_merge(base, overlay)
        assert merged == {
            "a": 1,
            "b": 2,
            "nested": {"x": 10, "y": 99, "z": 30},
        }

    def test_resolve_cascading_three_layers(self):
        job_layer = {
            "models": {"provider": "gemini", "model": "gemini-3.8-live", "temperature": 0.4},
            "instructions": {"system_prompt": "Base prompt"},
            "voice": {"voice_id": "Puck"},
        }
        room_layer = {
            "models": {"temperature": 0.7},  # Overrides temperature
            "vad": {"silence_duration_ms": 1800},
        }
        participant_layer = {
            "instructions": {
                "context_data": {"user": "Sarah"},  # Adds context data
            },
            "voice": {"voice_id": "Charon"},  # Overrides voice
        }

        merged = resolve_cascading_metadata(job_layer, room_layer, participant_layer)
        config = validate_metadata(merged)
        # Inherited from Job
        assert config.models.provider == "gemini"
        assert config.models.model == "gemini-3.8-live"
        assert config.instructions.system_prompt == "Base prompt"
        # Overridden by Room
        assert config.models.temperature == 0.7
        assert config.vad.silence_duration_ms == 1800
        # Overridden by Participant
        assert config.voice.voice_id == "Charon"
        assert config.instructions.context_data == {"user": "Sarah"}

    def test_extract_session_metadata_with_mock_job_context(self):
        job_metadata = json.dumps({
            "models": {"provider": "gemini", "model": "gemini-3.8-live"},
            "instructions": {"system_prompt": "Job level prompt"},
            "voice": {"voice_id": "Aoede"},
        })
        room_metadata = json.dumps({
            "instructions": {"system_prompt": "Room level prompt override"},
            "session": {"max_duration_seconds": 3600},
        })
        part_metadata = json.dumps({
            "session": {"inactivity_timeout_seconds": 90},
        })

        mock_participant = MagicMock()
        mock_participant.identity = "part_alpha"
        mock_participant.metadata = part_metadata
        mock_participant.attributes = {"role": "caller"}

        mock_job = MagicMock()
        mock_job.metadata = job_metadata

        mock_room = MagicMock()
        mock_room.metadata = room_metadata
        mock_room.remote_participants = {"part_alpha": mock_participant}

        mock_ctx = MagicMock(spec=JobContext)
        mock_ctx.job = mock_job
        mock_ctx.room = mock_room

        config = extract_session_metadata(mock_ctx)
        assert config.instructions.system_prompt == "Room level prompt override"
        assert config.session.max_duration_seconds == 3600
        assert config.session.inactivity_timeout_seconds == 90
        # Target identity automatically bound to remote participant
        assert config.session.target_participant_identity == "part_alpha"
        # Unmapped participant attributes routed to passthrough metadata
        assert config.metadata["role"] == "caller"

    def test_extract_session_metadata_participant_targeting(self):
        mock_p1 = MagicMock()
        mock_p1.identity = "part_observer"
        mock_p1.metadata = json.dumps({"voice": {"voice_id": "ObserverVoice"}})
        mock_p1.attributes = {}

        mock_p2 = MagicMock()
        mock_p2.identity = "target_speaker"
        mock_p2.metadata = json.dumps({"voice": {"voice_id": "TargetVoice"}})
        mock_p2.attributes = {}

        mock_job = MagicMock()
        mock_job.metadata = json.dumps({
            "models": {"provider": "gemini", "model": "gemini-3.8-live"},
            "instructions": {"system_prompt": "Targeted test"},
            "voice": {"voice_id": "DefaultVoice"},
            "session": {"target_participant_identity": "target_speaker"},
        })

        mock_room = MagicMock()
        mock_room.metadata = None
        mock_room.remote_participants = {
            "part_observer": mock_p1,
            "target_speaker": mock_p2,
        }

        mock_ctx = MagicMock(spec=JobContext)
        mock_ctx.job = mock_job
        mock_ctx.room = mock_room

        config = extract_session_metadata(mock_ctx)
        # Should pick up voice from target_speaker ("TargetVoice"), ignoring part_observer
        assert config.voice.voice_id == "TargetVoice"
        assert config.session.target_participant_identity == "target_speaker"


# ==============================================================================
# 6. Infrastructure Settings & Language Mapping Tests
# ==============================================================================
class TestInfrastructureSettings:
    """Verifies FrameworkSettings and language resolver functions."""

    def test_default_infrastructure_ports_and_timeouts(self):
        cfg = FrameworkSettings()
        assert cfg.health_check_port == 8081
        assert cfg.shutdown_timeout_seconds == 60.0
        assert cfg.livekit_url == "ws://127.0.0.1:7880"
        assert cfg.agent_name == "livekit-voice-agent"

    def test_bcp47_language_resolution(self):
        cfg = FrameworkSettings()
        assert cfg.resolve_bcp47_language("en") == "en-US"
        assert cfg.resolve_bcp47_language("english") == "en-US"
        assert cfg.resolve_bcp47_language("es-mx") == "es-MX"
        assert cfg.resolve_bcp47_language("spanish") == "es-US"
        assert cfg.resolve_bcp47_language("hi") == "hi-IN"
        assert cfg.resolve_bcp47_language(None) == "en-US"

    def test_asr_language_locking(self):
        cfg = FrameworkSettings()
        # Single language locking
        assert cfg.resolve_asr_language_codes("en") == ["en-US"]
        assert cfg.resolve_asr_language_codes("es") == ["es-US"]
        # Multi-language locking
        codes = cfg.resolve_asr_language_codes("en-US", extra_languages=["es-US", "fr-FR"])
        assert codes == ["en-US", "es-US", "fr-FR"]


# ==============================================================================
# 7. Hardened Validation, Edge Cases & Robustness Tests
# ==============================================================================
class TestHardenedValidationAndEdgeCases:
    """Verifies strict boundaries, tool schemas, auth pairings, and resolver edge cases."""

    def test_builtin_tool_unknown_name_rejected(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "unsupported_builtin",
            "type": "builtin",
        }]
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "Unsupported builtin tool" in str(exc.value)

    def test_builtin_tool_defaults(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "end_call",
            "type": "builtin",
        }]
        config = validate_metadata(payload)
        tool = config.tools[0]
        assert tool.name == "end_call"
        assert tool.config["delete_room"] is True
        assert len(tool.description) > 0

    def test_webhook_tool_invalid_method(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "update_record",
            "type": "webhook",
            "config": {
                "url": "https://api.example.com",
                "method": "PATCH",
            },
        }]
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "method" in str(exc.value)

    def test_webhook_tool_defaults(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "get_info",
            "type": "webhook",
            "config": {
                "url": "https://api.example.com/info",
            },
        }]
        config = validate_metadata(payload)
        tool = config.tools[0]
        assert tool.config["method"] == "POST"
        assert tool.config["timeout_seconds"] == 5.0
        assert tool.config["headers"] == {}

    def test_duplicate_tool_names_rejected(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [
            {"name": "fetch_data", "type": "webhook", "config": {"url": "https://a.com"}},
            {"name": "fetch_data", "type": "webhook", "config": {"url": "https://b.com"}},
        ]
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "Duplicate tool name" in str(exc.value)

    def test_first_turn_agent_listen_rejected(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "Agent prompt",
            "first_turn": {"speaker": "agent", "mode": "listen"},
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "Agent cannot initiate conversation" in str(exc.value)

    def test_first_turn_user_say_rejected(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "Agent prompt",
            "first_turn": {"speaker": "user", "mode": "say", "text": "Hello"},
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "User cannot be initiator" in str(exc.value)

    def test_first_turn_say_auto_defaults_agent(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["instructions"] = {
            "system_prompt": "Agent prompt",
            "first_turn": {"mode": "say", "text": "Hello there!"},
        }
        config = validate_metadata(payload)
        assert config.instructions.first_turn.speaker == "agent"
        assert config.instructions.first_turn.mode == "say"

    def test_voice_invalid_provider_rejected(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["voice"] = {"provider": "unsupported_tts", "voice_id": "v1"}
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "provider" in str(exc.value)

    def test_voice_language_normalization(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["voice"] = {"voice_id": "Aoede", "language": "english"}
        config = validate_metadata(payload)
        assert config.voice.language == "en-US"

    def test_modalities_validation(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["models"]["modalities"] = ["telepathy"]
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "modalities" in str(exc.value)

    def test_reporting_auth_unpaired_type_or_secret(self, minimal_valid_payload):
        # type given without secret_ref
        payload1 = dict(minimal_valid_payload)
        payload1["reporting"] = {
            "enabled": True,
            "endpoint": {"url": "https://api.example.com"},
            "auth": {"type": "bearer"},
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload1)
        assert "secret_ref" in str(exc.value)

        # secret_ref given without type
        payload2 = dict(minimal_valid_payload)
        payload2["reporting"] = {
            "enabled": True,
            "endpoint": {"url": "https://api.example.com"},
            "auth": {"secret_ref": "env:SECRET"},
        }
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload2)
        assert "type" in str(exc.value)

    def test_empty_agent_name_or_role(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["agent"] = {"name": "   ", "role": "assistant"}
        with pytest.raises(MetadataValidationError) as exc:
            validate_metadata(payload)
        assert "agent.name" in str(exc.value)

    def test_validate_metadata_session_config_with_secrets(self, minimal_valid_payload):
        payload = dict(minimal_valid_payload)
        payload["tools"] = [{
            "name": "lookup",
            "type": "webhook",
            "config": {
                "url": "https://api.example.com",
                "headers": {"Authorization": "Bearer {{env:AUTH_TOKEN}}"},
            },
        }]
        # First validate without resolving secrets
        config = validate_metadata(payload)
        assert "{{env:AUTH_TOKEN}}" in config.tools[0].config["headers"]["Authorization"]

        # Now pass existing SessionConfig with resolve_env_secrets=True
        mock_env = {"AUTH_TOKEN": "secret_abc_123"}
        resolved_config = validate_metadata(config, resolve_env_secrets=True, env=mock_env)
        assert resolved_config.tools[0].config["headers"]["Authorization"] == "Bearer secret_abc_123"

    def test_secrets_resolution_non_string_env_mapping(self):
        mock_env = {"PORT_NUM": 8080, "IS_ACTIVE": True}  # non-string values
        template = "http://localhost:{{env:PORT_NUM}}/status"
        res = resolve_secret(template, env=mock_env)
        assert res == "http://localhost:8080/status"

    def test_secrets_resolution_deeply_nested_structures(self):
        mock_env = {"SECRET_PARAM": "resolved_val_99"}
        nested_data = {
            "tools": [{
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "items": [{"default": "{{env:SECRET_PARAM}}"}],
                        }
                    },
                }
            }]
        }
        res = resolve_secrets(nested_data, env=mock_env, strict=True)
        assert res["tools"][0]["parameters"]["properties"]["filters"]["items"][0]["default"] == "resolved_val_99"

    def test_templating_json_serialization_of_nested_dict_and_list(self):
        prompt = "User profile: {profile}, permissions: {permissions}"
        ctx = {
            "profile": {"name": "Alice", "role": "admin"},
            "permissions": ["read", "write"],
        }
        interpolated = interpolate_prompt(prompt, ctx)
        assert '{"name": "Alice", "role": "admin"}' in interpolated
        assert '["read", "write"]' in interpolated

    def test_extract_session_metadata_preserves_target_when_no_participants_yet(self):
        mock_job = MagicMock()
        mock_job.metadata = json.dumps({
            "models": {"provider": "gemini", "model": "gemini-3.8-live"},
            "instructions": {"system_prompt": "Wait for user"},
            "voice": {"voice_id": "Aoede"},
        })
        mock_room = MagicMock()
        mock_room.metadata = None
        mock_room.remote_participants = {}  # No participants joined yet!

        mock_ctx = MagicMock(spec=JobContext)
        mock_ctx.job = mock_job
        mock_ctx.room = mock_room

        config = extract_session_metadata(mock_ctx, target_participant_identity="expected_caller_42")
        assert config.session.target_participant_identity == "expected_caller_42"

    def test_extract_participant_metadata_with_dict_and_json_array(self):
        mock_participant = MagicMock()
        mock_participant.metadata = None
        mock_participant.attributes = {
            "metadata": {"session": {"inactivity_timeout_seconds": 240}},  # Dict directly in attributes
            "tools": json.dumps([{"name": "end_call", "type": "builtin"}]),  # JSON array string
            "custom_vocabulary": json.dumps(["orthodontics", "cavity"]),
            "trainee_score": 95,
        }
        res = extract_participant_metadata(mock_participant)
        assert res["session"]["inactivity_timeout_seconds"] == 240
        assert isinstance(res["tools"], list)
        assert res["tools"][0]["name"] == "end_call"
        assert res["metadata"]["custom_vocabulary"] == ["orthodontics", "cavity"]
        assert res["metadata"]["trainee_score"] == 95

    def test_normalize_metadata_for_contract_with_nested_session_config(self):
        from agent_framework.config.resolver import normalize_metadata_for_contract

        raw_payload = {
            "scenario": "doctor-detailing",
            "patient_name": "Dr. Arthur Hayes",
            "trainee_id": "REP-101",
            "trainee_name": "Medical Representative",
            "session_config": {
                "version": "1.0",
                "agent": {"name": "ira-voice-agent", "role": "doctor"},
                "models": {"provider": "gemini", "model": "gemini-3.8-live"},
                "instructions": {"system_prompt": "You are Dr. Hayes, an interventional cardiologist."},
                "voice": {"voice_id": "Charon"},
            },
        }

        normalized = normalize_metadata_for_contract(raw_payload)
        # Verify nested session_config is elevated
        assert normalized["agent"]["name"] == "ira-voice-agent"
        assert normalized["agent"]["role"] == "doctor"
        assert normalized["models"]["model"] == "gemini-3.8-live"
        assert normalized["voice"]["voice_id"] == "Charon"
        # Verify outer unmapped fields are routed into metadata
        assert normalized["metadata"]["scenario"] == "doctor-detailing"
        assert normalized["metadata"]["patient_name"] == "Dr. Arthur Hayes"
        assert normalized["metadata"]["trainee_id"] == "REP-101"
        assert normalized["metadata"]["trainee_name"] == "Medical Representative"

        # Verify it passes strict validate_metadata without extra-fields error
        config = validate_metadata(normalized)
        assert config.agent.name == "ira-voice-agent"
        assert config.metadata["scenario"] == "doctor-detailing"
        assert config.metadata["patient_name"] == "Dr. Arthur Hayes"

    def test_extract_session_metadata_normalizes_legacy_fields_in_job_metadata(self):
        mock_job = MagicMock()
        mock_job.metadata = json.dumps({
            "scenario": "doctor-detailing",
            "persona": "doctor",
            "patient_name": "Dr. Elena Rostova",
            "trainee_id": "REP-202",
            "models": {"provider": "gemini", "model": "gemini-3.8-live"},
            "instructions": {"system_prompt": "You are Dr. Elena Rostova, an endocrinologist."},
            "voice": {"voice_id": "Aoede"},
        })
        mock_room = MagicMock()
        mock_room.metadata = None
        mock_room.remote_participants = {}

        mock_ctx = MagicMock(spec=JobContext)
        mock_ctx.job = mock_job
        mock_ctx.room = mock_room

        config = extract_session_metadata(mock_ctx)
        assert config.models.model == "gemini-3.8-live"
        assert config.instructions.system_prompt == "You are Dr. Elena Rostova, an endocrinologist."
        assert config.voice.voice_id == "Aoede"
        assert config.metadata["scenario"] == "doctor-detailing"
        assert config.metadata["patient_name"] == "Dr. Elena Rostova"

    def test_normalize_metadata_for_contract_with_serialized_json_session_config(self):
        from agent_framework.config.resolver import normalize_metadata_for_contract

        raw_payload = {
            "scenario": "doctor-detailing",
            "session_config": json.dumps({
                "version": "1.0",
                "models": {"provider": "gemini", "model": "gemini-3.8-live"},
                "instructions": {"system_prompt": "You are Dr. Hayes."},
                "voice": {"voice_id": "Charon"},
            }),
        }
        normalized = normalize_metadata_for_contract(raw_payload)
        assert normalized["models"]["model"] == "gemini-3.8-live"
        assert normalized["instructions"]["system_prompt"] == "You are Dr. Hayes."
        assert normalized["voice"]["voice_id"] == "Charon"
        assert normalized["metadata"]["scenario"] == "doctor-detailing"

        config = validate_metadata(normalized)
        assert config.models.model == "gemini-3.8-live"

    def test_normalize_metadata_for_contract_flat_legacy_parameters(self):
        from agent_framework.config.resolver import normalize_metadata_for_contract

        raw_payload = {
            "instructions": "You are Dr. Arthur Hayes, an interventional cardiologist.",
            "voice": "Charon",
            "model": "gemini-3.8-live",
            "temperature": 0.4,
            "first_turn_greeting": "Come in, please have a seat.",
            "vad_sensitivity_ms": 1500,
            "scenario": "doctor-detailing",
            "doctor_specialty": "Cardiology",
        }
        normalized = normalize_metadata_for_contract(raw_payload)
        assert normalized["instructions"]["system_prompt"] == "You are Dr. Arthur Hayes, an interventional cardiologist."
        assert normalized["instructions"]["first_turn"]["mode"] == "say"
        assert normalized["instructions"]["first_turn"]["text"] == "Come in, please have a seat."
        assert normalized["voice"]["voice_id"] == "Charon"
        assert normalized["models"]["model"] == "gemini-3.8-live"
        assert normalized["models"]["temperature"] == 0.4
        assert normalized["vad"]["silence_duration_ms"] == 1500
        assert normalized["metadata"]["scenario"] == "doctor-detailing"
        assert normalized["metadata"]["doctor_specialty"] == "Cardiology"

        config = validate_metadata(normalized)
        assert config.instructions.system_prompt == "You are Dr. Arthur Hayes, an interventional cardiologist."
        assert config.instructions.first_turn.text == "Come in, please have a seat."
        assert config.voice.voice_id == "Charon"

    def test_normalize_metadata_for_contract_full_doctor_detailing_log_payload(self):
        """Verifies exact real-world payload from frontend detailing scenario passes cleanly."""
        from agent_framework.config.resolver import normalize_metadata_for_contract

        raw_payload = {
            "scenario": "doctor-detailing",
            "persona": "doctor",
            "persona_name": "Dr. Arthur Hayes",
            "language": "en-US",
            "difficulty": "medium",
            "name": "Dr. Arthur Hayes",
            "patient_name": "Dr. Arthur Hayes",
            "customer_name": "Dr. Arthur Hayes",
            "trainee_id": "FDT-101",
            "trainee_name": "Front Desk Trainee",
            "gender": "male",
            "custom_instructions": "**Doctor Persona & Practice Profile**\nYou are Dr. Arthur Hayes, an interventional cardiologist.",
            "type": "Dr. Arthur Hayes — Interventional Cardiology",
            "optional_data": {"instructions": "**Doctor Persona**"},
            "agent_name": "ira-voice-agent",
            "planted_objection": "none",
            "scenario_id": "doctor-cardiology-default",
            "title": "Dr. Arthur Hayes — Interventional Cardiology",
            "model": "gemini-3.8-live",
            "temperature": 0.4,
            "thinking_level": "high",
            "vad_sensitivity_ms": 1500,
            "session_config": {
                "version": "1.0",
                "agent": {"name": "ira-voice-agent", "role": "doctor"},
                "trainee_id": "FDT-101",
            },
            "instructions": "**Doctor Persona & Practice Profile**\nYou are Dr. Arthur Hayes.",
            "voice": "Charon",
        }

        normalized = normalize_metadata_for_contract(raw_payload)
        config = validate_metadata(normalized)

        assert config.models.model == "gemini-3.8-live"
        assert config.models.temperature == 0.4
        assert config.models.thinking.enabled is True
        assert config.models.thinking.reasoning_effort == "high"
        assert config.voice.voice_id == "Charon"
        assert config.voice.language == "en-US"
        assert "Dr. Arthur Hayes" in config.instructions.system_prompt
        assert config.metadata["scenario"] == "doctor-detailing"
        assert config.metadata["persona"] == "doctor"
        assert config.metadata["trainee_id"] == "FDT-101"
