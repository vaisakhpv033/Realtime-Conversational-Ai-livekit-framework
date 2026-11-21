"""Dynamic Tool Registry for resolving and building agent tools from SessionConfig."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional
import weakref

from livekit.agents.job import JobContext

from agent_framework.config.schema import (
    SessionConfig,
    ToolConfig,
    register_builtin_tool_name,
    unregister_builtin_tool_name,
)
from agent_framework.tools.builtin.end_call import EndCallTool
from agent_framework.tools.webhook_tool import WebhookTool

logger = logging.getLogger("agent_framework.tools.registry")

BuiltinToolFactory = Callable[[ToolConfig, Optional[JobContext]], Any]


def _default_end_call_factory(config: ToolConfig, ctx: Optional[JobContext] = None) -> EndCallTool:
    """Default factory for the 'end_call' builtin tool."""
    cfg = config.config or {}
    farewell = (
        cfg.get("farewell_instruction")
        or cfg.get("end_instructions")
        or "Say goodbye to the user briefly before hanging up."
    )
    return EndCallTool(
        name=config.name,
        description=config.description or None,
        farewell_instruction=farewell,
        delete_room=cfg.get("delete_room", True),
        drain_timeout_seconds=cfg.get("drain_timeout_seconds", 5.0),
        job_ctx=ctx,
    )


class ToolRegistry:
    """Registry managing tool factories and resolving tools from SessionConfig.

    Supports:
    - Builtin tools (pre-registered 'end_call' and custom-registered builtins).
    - Webhook tools (dynamic external REST APIs).
    - Extensible custom builtin tool registration with automatic schema synchronization.
    - Unique tool name enforcement.
    """

    _global_builtin_factories: dict[str, BuiltinToolFactory] = {
        "end_call": _default_end_call_factory,
    }
    _instances: weakref.WeakSet[ToolRegistry] = weakref.WeakSet()

    def __init__(
        self,
        *,
        env: Optional[Mapping[str, str]] = None,
        allow_local: bool = False,
    ) -> None:
        """Initializes an instance-scoped ToolRegistry.

        Args:
            env: Optional environment mapping for secret substitution in tools.
            allow_local: Permits local loopback/private IPs for webhook tools (testing only).
        """
        self._builtin_factories: dict[str, BuiltinToolFactory] = dict(self._global_builtin_factories)
        self._env = env
        self._allow_local = allow_local
        self._instances.add(self)

    @classmethod
    def register_builtin_global(cls, name: str, factory: BuiltinToolFactory) -> None:
        """Registers a builtin tool factory globally across all registry instances."""
        clean_name = name.strip()
        cls._global_builtin_factories[clean_name] = factory
        register_builtin_tool_name(clean_name)
        logger.info(f"Registered global builtin tool factory: '{clean_name}'")

    @classmethod
    def unregister_builtin_global(cls, name: str) -> None:
        """Unregisters a global builtin tool factory."""
        clean_name = name.strip()
        cls._global_builtin_factories.pop(clean_name, None)
        if clean_name != "end_call":
            unregister_builtin_tool_name(clean_name)
        logger.info(f"Unregistered global builtin tool factory: '{clean_name}'")

    def register_builtin(self, name: str, factory: BuiltinToolFactory) -> None:
        """Registers a custom builtin tool factory on this registry instance."""
        clean_name = name.strip()
        self._builtin_factories[clean_name] = factory
        # Synchronize schema validation so ToolConfig allows this builtin name
        register_builtin_tool_name(clean_name)
        logger.info(f"Registered instance builtin tool factory: '{clean_name}'")

    def unregister_builtin(self, name: str) -> None:
        """Unregisters a builtin tool factory from this registry instance."""
        clean_name = name.strip()
        self._builtin_factories.pop(clean_name, None)
        still_in_use = any(clean_name in r._builtin_factories for r in self._instances)
        if not still_in_use and clean_name not in self._global_builtin_factories and clean_name != "end_call":
            unregister_builtin_tool_name(clean_name)
        logger.info(f"Unregistered instance builtin tool factory: '{clean_name}'")

    def is_builtin_registered(self, name: str) -> bool:
        """Checks if a builtin tool factory is registered."""
        return name.strip() in self._builtin_factories

    def get_registered_builtins(self) -> list[str]:
        """Returns the list of registered builtin tool names."""
        return sorted(self._builtin_factories.keys())

    def create_tool(
        self,
        tool_config: ToolConfig,
        ctx: Optional[JobContext] = None,
        *,
        allow_local: Optional[bool] = None,
    ) -> Any:
        """Instantiates a single tool from its ToolConfig model.

        Args:
            tool_config: Tool definition from SessionConfig.tools.
            ctx: Optional JobContext passed to builtin tools.
            allow_local: Optional override for allowing local loopback/private IPs.

        Returns:
            An instantiated LiveKit Tool or Toolset.

        Raises:
            ValueError: If the tool type is unknown or builtin tool is not registered.
        """
        tool_type = tool_config.type.strip().lower()
        effective_allow_local = self._allow_local if allow_local is None else allow_local

        if tool_type == "builtin":
            factory = self._builtin_factories.get(tool_config.name)
            if factory is None:
                raise ValueError(
                    f"Unknown builtin tool '{tool_config.name}'. "
                    f"Available builtin tools: {self.get_registered_builtins()}"
                )
            return factory(tool_config, ctx)

        elif tool_type == "webhook":
            return WebhookTool.from_config(
                tool_config,
                env=self._env,
                allow_local=effective_allow_local,
            )

        else:
            raise ValueError(f"Unsupported tool type '{tool_config.type}' for tool '{tool_config.name}'")

    def build_tools(
        self,
        session_config: SessionConfig,
        ctx: Optional[JobContext] = None,
    ) -> list[Any]:
        """Iterates through declared tools in SessionConfig and instantiates each one.

        Enforces unique tool names and returns the list of instantiated tools
        to pass into ``Agent(tools=...)``.

        Args:
            session_config: Validated SessionConfig containing declared tools.
            ctx: Optional LiveKit JobContext.

        Returns:
            list[Any]: List of instantiated Tool or Toolset instances.

        Raises:
            ValueError: If duplicate tool names are detected or a tool cannot be resolved.
        """
        allow_local = self._allow_local
        if not allow_local and hasattr(session_config, "reporting") and session_config.reporting:
            if hasattr(session_config.reporting, "security") and session_config.reporting.security:
                allow_local = session_config.reporting.security.allow_local

        instantiated_tools: list[Any] = []
        seen_names: set[str] = set()

        for tool_cfg in session_config.tools:
            name = tool_cfg.name.strip()
            if name in seen_names:
                raise ValueError(f"Duplicate tool name '{name}' detected in session tools configuration.")
            seen_names.add(name)

            tool_instance = self.create_tool(tool_cfg, ctx=ctx, allow_local=allow_local)
            instantiated_tools.append(tool_instance)

        logger.info(
            f"Successfully built {len(instantiated_tools)} tools for session: "
            f"{list(seen_names)}"
        )
        return instantiated_tools

    # Class-level convenience methods delegating to a default instance
    @classmethod
    def build_tools_default(
        cls,
        session_config: SessionConfig,
        ctx: Optional[JobContext] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        allow_local: bool = False,
    ) -> list[Any]:
        """Convenience class method to build tools using default settings."""
        registry = cls(env=env, allow_local=allow_local)
        return registry.build_tools(session_config, ctx=ctx)

    @classmethod
    def build_tools_from_config(
        cls,
        session_config: SessionConfig,
        ctx: Optional[JobContext] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        allow_local: bool = False,
    ) -> list[Any]:
        """Convenience class method to build tools from SessionConfig."""
        registry = cls(env=env, allow_local=allow_local)
        return registry.build_tools(session_config, ctx=ctx)


# Singleton default registry instance
default_tool_registry = ToolRegistry()


def create_tool(tool_config: ToolConfig, ctx: Optional[JobContext] = None) -> Any:
    """Convenience function creating a single tool via the default registry."""
    return default_tool_registry.create_tool(tool_config, ctx=ctx)
