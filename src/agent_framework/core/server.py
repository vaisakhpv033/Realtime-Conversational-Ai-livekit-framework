"""AgentServer configuration, health checks, and worker CLI runner.

Decouples the server runner and health-check infrastructure from application logic,
providing production-ready port 8081 health checks, shutdown timeouts, and entrypoint wiring.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from livekit.agents import AgentServer, JobContext, cli

from agent_framework.config.settings import FrameworkSettings, settings as default_settings
from agent_framework.core.session import SessionCoordinator

logger = logging.getLogger("agent_framework.core.server")


def create_server(
    settings: Optional[FrameworkSettings] = None,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    shutdown_process_timeout: Optional[float] = None,
    load_threshold: Optional[float] = None,
    drain_timeout: Optional[float] = None,
    prometheus_port: Optional[int] = None,
    log_level: Optional[str] = None,
    **kwargs: Any,
) -> AgentServer:
    """Creates a configured LiveKit AgentServer instance.

    Configures health checks (port 8081 by default), shutdown process timeouts,
    worker load threshold, graceful call draining, and Prometheus metrics.

    Args:
        settings: Optional custom FrameworkSettings; defaults to global settings.
        host: Host override for health checks / server.
        port: Port override for health check server (defaults to 8081).
        shutdown_process_timeout: Timeout in seconds for graceful process shutdown.
        load_threshold: CPU/memory worker load threshold.
        drain_timeout: Timeout in seconds for graceful active call draining before shutdown.
        prometheus_port: Optional port for Prometheus metrics exporter.
        log_level: Logging verbosity level for the LiveKit worker.
        **kwargs: Additional parameters forwarded to AgentServer.

    Returns:
        Configured AgentServer instance.
    """
    effective_settings = settings or default_settings

    resolved_host = host if host is not None else effective_settings.health_check_host
    resolved_port = port if port is not None else effective_settings.health_check_port
    resolved_shutdown_timeout = (
        shutdown_process_timeout
        if shutdown_process_timeout is not None
        else effective_settings.shutdown_timeout_seconds
    )
    resolved_load_threshold = (
        load_threshold
        if load_threshold is not None
        else effective_settings.worker_load_threshold
    )
    resolved_drain_timeout = (
        drain_timeout
        if drain_timeout is not None
        else effective_settings.drain_timeout_seconds
    )
    resolved_prometheus_port = (
        prometheus_port
        if prometheus_port is not None
        else effective_settings.prometheus_port
    )
    resolved_log_level = (
        log_level
        if log_level is not None
        else effective_settings.livekit_log_level
    )

    logger.info(
        f"Initializing AgentServer: host='{resolved_host}', port={resolved_port}, "
        f"shutdown_timeout={resolved_shutdown_timeout}s, drain_timeout={resolved_drain_timeout}s, "
        f"load_threshold={resolved_load_threshold}, log_level='{resolved_log_level}', "
        f"prometheus_port={resolved_prometheus_port}"
    )

    server_kwargs: dict[str, Any] = {
        "host": resolved_host,
        "port": resolved_port,
        "shutdown_process_timeout": resolved_shutdown_timeout,
        "load_threshold": resolved_load_threshold,
        "drain_timeout": int(resolved_drain_timeout),
        "log_level": resolved_log_level,
    }
    if "ws_url" not in kwargs and effective_settings.livekit_url:
        server_kwargs["ws_url"] = effective_settings.livekit_url
    if "api_key" not in kwargs and effective_settings.livekit_api_key:
        server_kwargs["api_key"] = effective_settings.livekit_api_key
    if "api_secret" not in kwargs and effective_settings.livekit_api_secret:
        server_kwargs["api_secret"] = effective_settings.livekit_api_secret
    if resolved_prometheus_port is not None:
        server_kwargs["prometheus_port"] = resolved_prometheus_port
    server_kwargs.update(kwargs)

    return AgentServer(**server_kwargs)


_active_coordinator: Optional[SessionCoordinator] = None


def get_active_coordinator() -> SessionCoordinator:
    """Returns the active SessionCoordinator instance, creating one if not yet initialized."""
    global _active_coordinator
    if _active_coordinator is None:
        _active_coordinator = SessionCoordinator()
    return _active_coordinator


def set_active_coordinator(coordinator: SessionCoordinator) -> None:
    """Explicitly sets the active SessionCoordinator instance."""
    global _active_coordinator
    _active_coordinator = coordinator


async def entrypoint(ctx: JobContext) -> None:
    """Top-level picklable RTC session entrypoint for LiveKit Agent worker processes."""
    coord = get_active_coordinator()
    await coord.run(ctx)


async def on_session_end(session_id: str) -> None:
    """Top-level picklable session-end callback for LiveKit Agent worker processes."""
    coord = get_active_coordinator()
    await coord.on_session_end(session_id)


def setup_server(
    server: Optional[AgentServer] = None,
    coordinator: Optional[SessionCoordinator] = None,
    settings: Optional[FrameworkSettings] = None,
    agent_name: Optional[str] = None,
) -> AgentServer:
    """Wires the SessionCoordinator entrypoint and session-end callback into an AgentServer.

    Args:
        server: Optional AgentServer; creates one via create_server() if None.
        coordinator: Optional SessionCoordinator instance; defaults to new SessionCoordinator().
        settings: Optional FrameworkSettings.
        agent_name: Logical agent name registering the entrypoint.

    Returns:
        AgentServer wired with the RTC session entrypoint.
    """
    effective_settings = settings or default_settings
    srv = server if server is not None else create_server(effective_settings)
    if coordinator is not None:
        set_active_coordinator(coordinator)
    effective_name = agent_name or effective_settings.agent_name

    srv.rtc_session(
        agent_name=effective_name,
        on_session_end=on_session_end,
    )(entrypoint)

    logger.info(f"Registered RTC session entrypoint for agent '{effective_name}'")
    return srv


def run_app(server: Optional[AgentServer] = None, **kwargs: Any) -> None:
    """Starts the LiveKit agent worker via CLI runner.

    Args:
        server: Optional AgentServer; creates and wires one if None.
        **kwargs: Additional parameters forwarded when initializing the server if server is None.
    """
    if server is not None:
        srv = server
    else:
        settings = kwargs.pop("settings", None)
        coordinator = kwargs.pop("coordinator", None)
        agent_name = kwargs.pop("agent_name", None)
        created_srv = create_server(settings=settings, **kwargs)
        srv = setup_server(
            server=created_srv,
            coordinator=coordinator,
            settings=settings,
            agent_name=agent_name,
        )
    logger.info("Starting LiveKit Agent worker via CLI runner...")
    cli.run_app(srv)


# Alias for run_app
run_server = run_app


def main() -> None:
    """Entry point for CLI scripts."""
    run_app()


if __name__ == "__main__":
    main()

