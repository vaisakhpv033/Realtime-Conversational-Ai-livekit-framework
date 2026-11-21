"""Configuration, metadata schemas, strict validation, and resolution engine."""

from agent_framework.config.resolver import (
    deep_merge,
    extract_participant_metadata,
    extract_session_metadata,
    resolve_cascading_metadata,
)
from agent_framework.config.schema import (
    AgentConfig,
    AgentMetadataContract,
    FirstTurnConfig,
    InstructionsConfig,
    ModelsConfig,
    RecordingConfig,
    ReportingAuthConfig,
    ReportingConfig,
    ReportingEndpointConfig,
    ReportingPayloadConfig,
    ReportingSecurityConfig,
    RetryPolicyConfig,
    SessionConfig,
    SessionLifecycleConfig,
    ThinkingConfig,
    ToolConfig,
    TurnDetectionConfig,
    VADConfig,
    VoiceConfig,
    get_effective_system_prompt,
    register_builtin_tool_name,
    unregister_builtin_tool_name,
)
from agent_framework.config.settings import (
    LANGUAGE_BCP47_MAP,
    FrameworkSettings,
    settings,
)
from agent_framework.config.validator import (
    MetadataValidationError,
    validate_metadata,
)

__all__ = [
    # Schema models
    "AgentConfig",
    "AgentMetadataContract",
    "FirstTurnConfig",
    "InstructionsConfig",
    "ModelsConfig",
    "RecordingConfig",
    "ReportingAuthConfig",
    "ReportingConfig",
    "ReportingEndpointConfig",
    "ReportingPayloadConfig",
    "ReportingSecurityConfig",
    "RetryPolicyConfig",
    "SessionConfig",
    "SessionLifecycleConfig",
    "ThinkingConfig",
    "ToolConfig",
    "TurnDetectionConfig",
    "VADConfig",
    "VoiceConfig",
    "get_effective_system_prompt",
    "register_builtin_tool_name",
    "unregister_builtin_tool_name",
    # Validator
    "MetadataValidationError",
    "validate_metadata",
    # Resolver
    "deep_merge",
    "extract_participant_metadata",
    "extract_session_metadata",
    "resolve_cascading_metadata",
    # Settings
    "FrameworkSettings",
    "LANGUAGE_BCP47_MAP",
    "settings",
]
