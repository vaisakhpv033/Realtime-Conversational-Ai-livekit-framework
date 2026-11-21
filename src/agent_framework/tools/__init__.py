"""Dynamic Tool Registry and LiveKit Function Calling Broker.

Provides extensible tooling for voice agents:
- EndCallTool: Configurable call termination with utterance-drain synchronization and room teardown.
- WebhookTool: Dynamic REST API caller executing external endpoints with SSRF guards.
- ToolRegistry: Factory and registry resolving declared tools from SessionConfig.
"""

from agent_framework.tools.builtin.end_call import (
    DEFAULT_END_CALL_DESCRIPTION,
    DEFAULT_FAREWELL_INSTRUCTION,
    EndCallTool,
)
from agent_framework.tools.registry import (
    ToolRegistry,
    create_tool,
    default_tool_registry,
)
from agent_framework.tools.webhook_tool import (
    SSRFSecurityError,
    WebhookConnectionError,
    WebhookHTTPError,
    WebhookTimeoutError,
    WebhookTool,
    WebhookToolError,
)

__all__ = [
    # Tool Registry
    "ToolRegistry",
    "create_tool",
    "default_tool_registry",
    # Builtin Tools
    "EndCallTool",
    "DEFAULT_END_CALL_DESCRIPTION",
    "DEFAULT_FAREWELL_INSTRUCTION",
    # Webhook Tool
    "WebhookTool",
    "WebhookToolError",
    "SSRFSecurityError",
    "WebhookHTTPError",
    "WebhookTimeoutError",
    "WebhookConnectionError",
]
