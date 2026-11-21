"""AutoCoach backward-compatibility bridge to agent_framework."""

from agent_framework import (
    SessionConfig,
    SessionContext,
    SessionCoordinator,
    attach_to_job,
    detach_from_job,
    get_session_context,
    create_server,
    setup_server,
    run_server,
    run_app,
    main,
)

__version__ = "1.0.0"

__all__ = [
    "SessionConfig",
    "SessionContext",
    "SessionCoordinator",
    "attach_to_job",
    "detach_from_job",
    "get_session_context",
    "create_server",
    "setup_server",
    "run_server",
    "run_app",
    "main",
]
