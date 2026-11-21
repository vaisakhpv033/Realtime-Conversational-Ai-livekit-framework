"""Strict Pydantic V2 metadata contract and schema definitions.

Defines schemas for all 10 metadata configuration sections:
1. agent
2. models
3. instructions
4. voice
5. vad
6. turn_detection
7. session
8. tools
9. reporting
10. metadata
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# 1. Agent Configuration
# ==============================================================================
class AgentConfig(BaseModel):
    """Identity and telemetry context for worker logs and metrics."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="voice-agent",
        description="Logical name for worker logs and telemetry",
    )
    role: str = Field(
        default="assistant",
        description="Agent role (e.g., 'assistant', 'receptionist', 'triage_specialist', 'interviewer')",
    )
    description: Optional[str] = Field(
        default=None,
        description="High-level description of agent intent for logging/metrics",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("agent.name cannot be empty or whitespace only")
        return clean

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("agent.role cannot be empty or whitespace only")
        return clean


# ==============================================================================
# 2. Models Configuration
# ==============================================================================
class ThinkingConfig(BaseModel):
    """Reasoning and thinking budget settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enables extended thinking/reasoning",
    )
    reasoning_effort: str = Field(
        default="medium",
        description="Thinking level: 'minimal', 'low', 'medium', 'high'",
    )
    budget_tokens: int = Field(
        default=1024,
        ge=0,
        description="Explicit token budget for thinking models",
    )
    include_thoughts: bool = Field(
        default=False,
        description="Whether thought traces are surfaced in events",
    )

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        clean = v.strip().lower()
        valid = {"minimal", "low", "medium", "high"}
        if clean not in valid:
            raise ValueError(f"Invalid reasoning_effort '{v}'. Must be one of: {sorted(valid)}")
        return clean


class ModelsConfig(BaseModel):
    """Model provider, hyperparameters, and dynamic ASR vocabulary."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        ...,
        min_length=1,
        description="Model provider enum ('gemini', 'openai', 'pipeline')",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier (e.g. 'gemini-3.8-live', 'gemini-3.8-live-extended-thinking')",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model sampling temperature",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold",
    )
    top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling limit",
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max output generation tokens",
    )
    modalities: list[str] = Field(
        default_factory=lambda: ["audio"],
        description="Supported modalities (['audio'], ['audio', 'text'])",
    )
    thinking: ThinkingConfig = Field(
        default_factory=ThinkingConfig,
        description="Thinking and reasoning configuration",
    )
    custom_vocabulary: list[str] = Field(
        default_factory=list,
        description="Dynamic ASR speech-recognition biasing terms for the session",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        clean = v.strip().lower()
        valid = {"gemini", "openai", "pipeline"}
        if clean not in valid:
            raise ValueError(f"Invalid provider '{v}'. Must be one of: {sorted(valid)}")
        return clean

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("model identifier cannot be empty")
        return clean

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("modalities cannot be empty")
        valid = {"audio", "text"}
        cleaned: list[str] = []
        for item in v:
            c = item.strip().lower()
            if c not in valid:
                raise ValueError(f"Invalid modality '{item}'. Must be one of: {sorted(valid)}")
            if c not in cleaned:
                cleaned.append(c)
        return cleaned


# ==============================================================================
# 3. Instructions Configuration
# ==============================================================================
class FirstTurnConfig(BaseModel):
    """First-turn initiation rules and greeting behavior."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(
        default="user",
        description="Who initiates the conversation: 'agent' or 'user'",
    )
    mode: str = Field(
        default="listen",
        description="Initial agent turn mode: 'say', 'generate_reply', 'listen'",
    )
    text: Optional[str] = Field(
        default=None,
        description="Initial greeting string if mode='say'",
    )

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"agent", "user"}:
            raise ValueError("speaker must be 'agent' or 'user'")
        return clean

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"say", "generate_reply", "listen"}:
            raise ValueError("mode must be 'say', 'generate_reply', or 'listen'")
        return clean

    @model_validator(mode="before")
    @classmethod
    def adjust_speaker_and_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If mode is 'say' or 'generate_reply' and speaker was not specified, default speaker to 'agent'
            if data.get("mode") in {"say", "generate_reply"} and "speaker" not in data:
                data["speaker"] = "agent"
            # If speaker is 'agent' and mode not specified: default to 'say' if text given, else 'generate_reply'
            elif data.get("speaker") == "agent" and "mode" not in data:
                data["mode"] = "say" if data.get("text") else "generate_reply"
        return data

    @model_validator(mode="after")
    def validate_first_turn_logic(self) -> Self:
        if self.mode == "say" and (not self.text or not self.text.strip()):
            raise ValueError("first_turn.text must be provided and non-empty when mode is 'say'")
        if self.speaker == "agent" and self.mode == "listen":
            raise ValueError(
                "Agent cannot initiate conversation (speaker='agent') in 'listen' mode. Mode must be 'say' or 'generate_reply'."
            )
        if self.speaker == "user" and self.mode == "say":
            raise ValueError("User cannot be initiator (speaker='user') when agent first turn mode is 'say'.")
        return self


class InstructionsConfig(BaseModel):
    """System prompts, context interpolation, and conversational dialogue rules."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(
        ...,
        min_length=1,
        description="Core instructions driving agent behavior",
    )
    first_turn: FirstTurnConfig = Field(
        default_factory=FirstTurnConfig,
        description="First-turn greeting and initiation settings",
    )
    context_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs interpolated into {key} placeholders in system_prompt",
    )
    spoken_dialogue_rules: bool = Field(
        default=True,
        description="Appends universal, domain-neutral rules for concise spoken conversation",
    )

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("system_prompt cannot be empty or whitespace only")
        if len(v) > 65536:
            raise ValueError(f"system_prompt exceeds maximum length of 65536 characters (got {len(v)})")
        return v

    def get_effective_prompt(self) -> str:
        """Returns the final system prompt with context interpolation and spoken dialogue rules applied."""
        from agent_framework.utils.templating import interpolate_prompt

        prompt = interpolate_prompt(self.system_prompt, self.context_data)
        if self.spoken_dialogue_rules and "Conversational speech guidelines:" not in prompt:
            prompt = (
                prompt.rstrip()
                + "\n\n"
                + "Conversational speech guidelines:\n"
                + "- Speak conversationally, naturally, and concisely (typically 1-2 sentences per turn).\n"
                + "- Do not format responses with markdown, bullet points, asterisks, or tables, as this will be spoken aloud.\n"
                + "- Ask clarifying questions when needed, but keep them focused and brief."
            )
        return prompt

    def get_effective_system_prompt(self) -> str:
        """Alias for get_effective_prompt."""
        return self.get_effective_prompt()


# ==============================================================================
# 4. Voice Configuration
# ==============================================================================
class VoiceConfig(BaseModel):
    """Acoustic, voice engine, and language settings."""

    model_config = ConfigDict(extra="forbid")

    provider: Optional[str] = Field(
        default=None,
        description="Voice engine ('gemini', 'openai', 'elevenlabs'). Defaults to models.provider if None.",
    )
    voice_id: str = Field(
        ...,
        min_length=1,
        description="Provider voice identifier (e.g. 'Aoede', 'Puck', 'alloy')",
    )
    language: str = Field(
        default="en-US",
        description="BCP-47 language tag for synthesis and ASR language locking",
    )
    speed: float = Field(
        default=1.0,
        gt=0.0,
        le=4.0,
        description="Speech playback speed multiplier",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().lower()
        valid = {"gemini", "openai", "elevenlabs", "pipeline"}
        if clean not in valid:
            raise ValueError(f"Invalid voice provider '{v}'. Must be one of: {sorted(valid)}")
        return clean

    @field_validator("voice_id")
    @classmethod
    def validate_voice_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("voice_id cannot be empty or whitespace only")
        return clean

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("language cannot be empty")
        from agent_framework.config.settings import settings
        return settings.resolve_bcp47_language(clean)


# ==============================================================================
# 5. VAD Configuration
# ==============================================================================
class VADConfig(BaseModel):
    """Voice Activity Detection parameters."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default="native",
        description="VAD engine ('native' server VAD or 'silero')",
    )
    disabled: bool = Field(
        default=False,
        description="Disables automatic voice activity detection",
    )
    silence_duration_ms: int = Field(
        default=1500,
        ge=200,
        le=3000,
        description="Silence duration in milliseconds before turn endpointing",
    )
    prefix_padding_ms: int = Field(
        default=300,
        ge=0,
        le=2000,
        description="Audio prepended before speech onset in milliseconds",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice detection probability threshold",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"native", "silero"}:
            raise ValueError(f"Invalid vad provider '{v}'. Must be 'native' or 'silero'")
        return clean


# ==============================================================================
# 6. Turn Detection Configuration
# ==============================================================================
class TurnDetectionConfig(BaseModel):
    """Conversational dynamics, interruption handling, and endpointing rules."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        default="server_vad",
        description="Turn detection mode",
    )
    allow_interruptions: bool = Field(
        default=True,
        description="Whether user can interrupt agent speech",
    )
    min_interruption_duration_ms: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Duration of user speech before triggering interruption",
    )
    min_interruption_words: int = Field(
        default=2,
        ge=0,
        description="Word count threshold before interrupting",
    )
    resume_false_interruption: bool = Field(
        default=True,
        description="Resumes speech if user utterance was just a backchannel ('mhm', 'yeah')",
    )
    min_endpointing_delay_ms: int = Field(
        default=500,
        ge=0,
        le=5000,
        description="Minimum silence before finalizing turn",
    )
    user_away_timeout_seconds: float = Field(
        default=20.0,
        gt=0.0,
        le=300.0,
        description="Timeout before triggering user-inactivity handling",
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_nested_turn_options(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            interruption = data.pop("interruption", None)
            if isinstance(interruption, dict):
                if "enabled" in interruption and "allow_interruptions" not in data:
                    data["allow_interruptions"] = interruption["enabled"]
                if "resume_false_interruption" in interruption and "resume_false_interruption" not in data:
                    data["resume_false_interruption"] = interruption["resume_false_interruption"]
                if "min_duration_ms" in interruption and "min_interruption_duration_ms" not in data:
                    data["min_interruption_duration_ms"] = interruption["min_duration_ms"]
                elif "min_duration" in interruption and "min_interruption_duration_ms" not in data:
                    val = interruption["min_duration"]
                    data["min_interruption_duration_ms"] = int(val * 1000) if isinstance(val, (int, float)) and val < 10 else int(val)
                if "min_words" in interruption and "min_interruption_words" not in data:
                    data["min_interruption_words"] = interruption["min_words"]

            endpointing = data.pop("endpointing", None)
            if isinstance(endpointing, dict):
                if "min_delay_ms" in endpointing and "min_endpointing_delay_ms" not in data:
                    data["min_endpointing_delay_ms"] = endpointing["min_delay_ms"]
                elif "min_delay" in endpointing and "min_endpointing_delay_ms" not in data:
                    val = endpointing["min_delay"]
                    data["min_endpointing_delay_ms"] = int(val * 1000) if isinstance(val, (int, float)) and val < 10 else int(val)
        return data


# ==============================================================================
# 7. Session Lifecycle Configuration
# ==============================================================================
class RecordingConfig(BaseModel):
    """Native LiveKit session recording settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enables native LiveKit session recording",
    )
    audio: bool = Field(
        default=True,
        description="Records audio track",
    )
    traces: bool = Field(
        default=True,
        description="Records execution and event traces",
    )


class SessionLifecycleConfig(BaseModel):
    """Session watchdogs, participant binding, and cleanup policies."""

    model_config = ConfigDict(extra="forbid")

    max_duration_seconds: int = Field(
        default=1800,
        ge=30,
        le=7200,
        description="Hard cap on call length in seconds",
    )
    inactivity_timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="Disconnects room if no interaction occurs for this duration",
    )
    participant_join_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=600.0,
        description="Max wait for participant before terminating job",
    )
    delete_room_on_terminate: bool = Field(
        default=True,
        description="Automatically closes the LiveKit room on disconnect",
    )
    target_participant_identity: Optional[str] = Field(
        default=None,
        description="Specific participant identity to bind with in multi-party rooms",
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig,
        description="Session recording configuration",
    )


# ==============================================================================
# 8. Tools Configuration
# ==============================================================================
SUPPORTED_BUILTIN_TOOL_NAMES: set[str] = {"end_call"}


def register_builtin_tool_name(name: str) -> None:
    """Registers a builtin tool name so it passes schema validation."""
    SUPPORTED_BUILTIN_TOOL_NAMES.add(name.strip())


def unregister_builtin_tool_name(name: str) -> None:
    """Unregisters a builtin tool name from schema validation."""
    SUPPORTED_BUILTIN_TOOL_NAMES.discard(name.strip())


class ToolConfig(BaseModel):
    """Dynamic function calling declaration for builtin or webhook tools."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Tool name exposed to LLM",
    )
    type: str = Field(
        default="webhook",
        description="Tool type: 'builtin' or 'webhook'",
    )
    description: str = Field(
        default="",
        description="Tool prompt instructions for LLM",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Standard JSON Schema describing function arguments",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool configuration parameters (e.g. url, method, headers for webhook)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Tool name cannot be empty")
        return clean

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in {"builtin", "webhook"}:
            raise ValueError("Tool type must be 'builtin' or 'webhook'")
        return clean

    @model_validator(mode="after")
    def validate_tool_config(self) -> Self:
        if self.type == "builtin":
            if self.name not in SUPPORTED_BUILTIN_TOOL_NAMES:
                raise ValueError(
                    f"Unsupported builtin tool '{self.name}'. Currently supported builtin tools: {sorted(SUPPORTED_BUILTIN_TOOL_NAMES)}"
                )
            if self.name == "end_call":
                if not self.description:
                    self.description = (
                        "Gracefully concludes the call when the conversation is finished or user requests to hang up."
                    )
                # Default delete_room to True if not provided
                if "delete_room" not in self.config:
                    self.config["delete_room"] = True
                elif not isinstance(self.config["delete_room"], bool):
                    raise ValueError("Builtin tool config 'delete_room' must be a boolean")

                if "drain_timeout_seconds" in self.config:
                    drain = self.config["drain_timeout_seconds"]
                    if not isinstance(drain, (int, float)) or drain <= 0:
                        raise ValueError("Builtin tool config 'drain_timeout_seconds' must be a positive number")

        elif self.type == "webhook":
            url = self.config.get("url")
            if not url or not str(url).strip():
                raise ValueError(f"Webhook tool '{self.name}' must specify a non-empty 'url' in its config")

            # Default method to POST and validate
            method = self.config.setdefault("method", "POST")
            if not isinstance(method, str) or method.strip().upper() not in {"GET", "POST", "PUT"}:
                raise ValueError(
                    f"Webhook tool '{self.name}' config.method must be 'GET', 'POST', or 'PUT', got '{method}'"
                )
            self.config["method"] = method.strip().upper()

            # Default headers to dict
            headers = self.config.setdefault("headers", {})
            if not isinstance(headers, dict):
                raise ValueError(f"Webhook tool '{self.name}' config.headers must be a dictionary")

            # Default timeout_seconds to 5.0
            timeout = self.config.setdefault("timeout_seconds", 5.0)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(f"Webhook tool '{self.name}' config.timeout_seconds must be a positive number")

        return self


# ==============================================================================
# 9. Reporting Configuration
# ==============================================================================
class ReportingEndpointConfig(BaseModel):
    """Reporting target webhook endpoint settings."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        ...,
        min_length=1,
        description="Target webhook URL",
    )
    method: str = Field(
        default="POST",
        description="HTTP method ('POST', 'PUT')",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers (supports {{env:KEY}} secret references)",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=120.0,
        description="HTTP request timeout in seconds",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in {"POST", "PUT"}:
            raise ValueError("Reporting endpoint method must be 'POST' or 'PUT'")
        return clean

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Reporting endpoint url cannot be empty")
        return clean


class ReportingAuthConfig(BaseModel):
    """Authentication settings for reporting webhooks."""

    model_config = ConfigDict(extra="forbid")

    type: Optional[str] = Field(
        default=None,
        description="Auth type: 'bearer', 'api_key', 'hmac_sha256'",
    )
    secret_ref: Optional[str] = Field(
        default=None,
        description="Secret reference (e.g. 'env:REPORTING_SECRET')",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().lower()
        if clean not in {"bearer", "api_key", "hmac_sha256"}:
            raise ValueError("Reporting auth type must be 'bearer', 'api_key', or 'hmac_sha256'")
        return clean

    @model_validator(mode="after")
    def validate_auth_pairing(self) -> Self:
        if self.type and not self.secret_ref:
            raise ValueError(f"Reporting auth type '{self.type}' requires a non-empty 'secret_ref'")
        if self.secret_ref and not self.type:
            raise ValueError("Reporting auth 'secret_ref' provided without an auth 'type'")
        return self


class RetryPolicyConfig(BaseModel):
    """Retry policy for webhook dispatch."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    initial_delay_ms: int = Field(
        default=1000,
        ge=0,
        le=60000,
        description="Initial delay in milliseconds",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier",
    )


class ReportingSecurityConfig(BaseModel):
    """SSRF and local network access controls for reporting webhooks."""

    model_config = ConfigDict(extra="forbid")

    validate_ssrf: bool = Field(
        default=True,
        description="Blocks private IPs, loopbacks, and cloud metadata",
    )
    allow_local: bool = Field(
        default=False,
        description="Permits localhost/private IPs (testing only)",
    )
    allowed_domains: Optional[list[str]] = Field(
        default=None,
        description="Optional domain whitelist for webhooks",
    )


class ReportingPayloadConfig(BaseModel):
    """Selection of telemetry components included in the reporting payload."""

    model_config = ConfigDict(extra="forbid")

    include_chat_history: bool = Field(
        default=True,
        description="Whether to include chat transcripts",
    )
    include_model_usage: bool = Field(
        default=True,
        description="Reports token usage metrics",
    )
    include_events: bool = Field(
        default=True,
        description="Reports granular timeline events",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-specified passthrough fields",
    )


class ReportingConfig(BaseModel):
    """Comprehensive post-session telemetry and reporting webhook configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether post-session reporting is enabled",
    )
    endpoint: Optional[ReportingEndpointConfig] = Field(
        default=None,
        description="Target webhook endpoint",
    )
    auth: Optional[ReportingAuthConfig] = Field(
        default=None,
        description="Webhook authentication configuration",
    )
    retry_policy: RetryPolicyConfig = Field(
        default_factory=RetryPolicyConfig,
        description="Retry policy",
    )
    security: ReportingSecurityConfig = Field(
        default_factory=ReportingSecurityConfig,
        description="SSRF security settings",
    )
    payload_config: ReportingPayloadConfig = Field(
        default_factory=ReportingPayloadConfig,
        description="Payload formatting configuration",
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_enabled(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If enabled is not explicitly given, but endpoint with url is provided, default to True
            if "enabled" not in data:
                endpoint = data.get("endpoint")
                if isinstance(endpoint, dict) and endpoint.get("url"):
                    data["enabled"] = True
        return data

    @model_validator(mode="after")
    def validate_enabled_endpoint(self) -> Self:
        if self.enabled:
            if not self.endpoint or not self.endpoint.url:
                raise ValueError("reporting.endpoint.url is required when reporting is enabled")
        return self


# ==============================================================================
# 10. Top-Level Session Configuration (Contract)
# ==============================================================================
class SessionConfig(BaseModel):
    """The complete versioned metadata contract for LiveKit voice agent sessions.

    Controls all operational behaviors: agent identity, model hyperparameters,
    system prompt, voice settings, VAD thresholds, turn taking, watchdogs, tools,
    and post-call reporting.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        default="1.0",
        description="Contract schema version",
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent identity and telemetry metadata",
    )
    models: ModelsConfig = Field(
        ...,
        description="Model provider, hyperparameters, and thinking configuration",
    )
    instructions: InstructionsConfig = Field(
        ...,
        description="System prompts, first turn, and context data",
    )
    voice: VoiceConfig = Field(
        ...,
        description="Voice engine, voice ID, and speech speed",
    )
    vad: VADConfig = Field(
        default_factory=VADConfig,
        description="Voice Activity Detection parameters",
    )
    turn_detection: TurnDetectionConfig = Field(
        default_factory=TurnDetectionConfig,
        description="Turn detection and interruption parameters",
    )
    session: SessionLifecycleConfig = Field(
        default_factory=SessionLifecycleConfig,
        description="Session watchdog timeouts and cleanup policies",
    )
    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Dynamic function calling declarations",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Post-session reporting webhook configuration",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pass-through metadata echoed in telemetry",
    )

    @field_validator("tools")
    @classmethod
    def validate_unique_tool_names(cls, tools: list[ToolConfig]) -> list[ToolConfig]:
        seen: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                raise ValueError(f"Duplicate tool name '{tool.name}' found in tools configuration.")
            seen.add(tool.name)
        return tools

    @model_validator(mode="after")
    def apply_cross_field_defaults(self) -> Self:
        # Default voice.provider to models.provider if not explicitly set
        if self.voice and not self.voice.provider and self.models:
            self.voice.provider = self.models.provider
        return self

    def get_effective_system_prompt(self) -> str:
        """Convenience method returning the effective interpolated system prompt."""
        return self.instructions.get_effective_prompt()


# Backward-compatible alias
AgentMetadataContract = SessionConfig


def get_effective_system_prompt(instructions: InstructionsConfig | SessionConfig | str) -> str:
    """Helper returning the effective system prompt from InstructionsConfig, SessionConfig, or string."""
    if isinstance(instructions, SessionConfig):
        return instructions.get_effective_system_prompt()
    elif isinstance(instructions, InstructionsConfig):
        return instructions.get_effective_prompt()
    return str(instructions)
