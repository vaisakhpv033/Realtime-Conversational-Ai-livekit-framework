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
    **kwargs: Any,
) -> AgentServer:
    """Creates a configured LiveKit AgentServer instance.

    Configures health checks (port 8081 by default), shutdown process timeouts,
    and worker load threshold.

    Args:
        settings: Optional custom FrameworkSettings; defaults to global settings.
        host: Host override for health checks / server.
        port: Port override for health check server (defaults to 8081).
        shutdown_process_timeout: Timeout in seconds for graceful process shutdown.
        load_threshold: CPU/memory worker load threshold.
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

    logger.info(
        f"Initializing AgentServer: host='{resolved_host}', port={resolved_port}, "
        f"shutdown_timeout={resolved_shutdown_timeout}s, load_threshold={resolved_load_threshold}"
    )

    return AgentServer(
        host=resolved_host,
        port=resolved_port,
        shutdown_process_timeout=resolved_shutdown_timeout,
        load_threshold=resolved_load_threshold,
        **kwargs,
    )


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
    coord = coordinator if coordinator is not None else SessionCoordinator()
    effective_name = agent_name or effective_settings.agent_name

    @srv.rtc_session(
        agent_name=effective_name,
        on_session_end=coord.on_session_end,
    )
    async def entrypoint(ctx: JobContext) -> None:
        await coord.run(ctx)

    logger.info(f"Registered RTC session entrypoint for agent '{effective_name}'")
    return srv


def run_app(server: Optional[AgentServer] = None, **kwargs: Any) -> None:
    """Starts the LiveKit agent worker via CLI runner.

    Args:
        server: Optional AgentServer; creates and wires one if None.
        **kwargs: Additional parameters forwarded to cli.run_app.
    """
    srv = server if server is not None else setup_server()
    logger.info("Starting LiveKit Agent worker via CLI runner...")
    cli.run_app(srv, **kwargs)


# Alias for run_app
run_server = run_app


def main() -> None:
    """Entry point for CLI scripts."""
    run_app()


if __name__ == "__main__":
    main()

