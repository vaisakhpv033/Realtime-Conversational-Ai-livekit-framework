"""Safe secret resolution utility.

Resolves {{env:KEY}} and env:KEY references from environment variables
or custom secret mappings with strict fail-fast validation.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping


ENV_SECRET_PATTERN = re.compile(r"\{\{\s*env:([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
PREFIX_SECRET_PATTERN = re.compile(r"^env:([a-zA-Z_][a-zA-Z0-9_]*)$")


class SecretResolutionError(Exception):
    """Raised when a referenced environment secret cannot be resolved."""

    def __init__(self, variable_name: str, message: str | None = None) -> None:
        self.variable_name = variable_name
        msg = message or f"Environment variable '{variable_name}' referenced in secret template was not found."
        super().__init__(msg)


def resolve_secret(
    value: str,
    env: Mapping[str, str] | None = None,
    strict: bool = True,
) -> str:
    """Resolves secret references in a single string.

    Supports:
    - Mustache template syntax: ``Bearer {{env:API_KEY}}``
    - Direct prefix syntax: ``env:REPORTING_SECRET``

    Args:
        value: String containing literal text and/or secret references.
        env: Mapping of environment variables (defaults to os.environ).
        strict: If True, raises SecretResolutionError when a referenced env var is missing.
                If False, leaves the placeholder untouched.

    Returns:
        The string with secret references replaced with their resolved values.

    Raises:
        SecretResolutionError: When strict=True and a referenced variable is missing.
    """
    if not isinstance(value, str) or not value:
        return value

    target_env = env if env is not None else os.environ

    # Check for direct prefix match: "env:VAR_NAME"
    prefix_match = PREFIX_SECRET_PATTERN.match(value.strip())
    if prefix_match:
        var_name = prefix_match.group(1)
        if var_name in target_env:
            return str(target_env[var_name])
        if strict:
            raise SecretResolutionError(var_name)
        return value

    # Check for template placeholders: "{{env:VAR_NAME}}"
    def _replace_match(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in target_env:
            return str(target_env[var_name])
        if strict:
            raise SecretResolutionError(var_name)
        return match.group(0)

    return ENV_SECRET_PATTERN.sub(_replace_match, value)


def resolve_secrets(
    data: Any,
    env: Mapping[str, str] | None = None,
    strict: bool = True,
) -> Any:
    """Recursively resolves secret references in nested data structures.

    Processes dictionaries (both keys and values if strings), lists,
    and strings. Non-string primitives (int, float, bool, None) are returned as-is.

    Args:
        data: Arbitrary data structure to resolve.
        env: Mapping of environment variables (defaults to os.environ).
        strict: If True, raises SecretResolutionError when any referenced env var is missing.

    Returns:
        A new data structure with all secret references resolved.
    """
    if isinstance(data, str):
        return resolve_secret(data, env=env, strict=strict)

    if isinstance(data, dict):
        resolved_dict: dict[Any, Any] = {}
        for k, v in data.items():
            resolved_k = resolve_secret(k, env=env, strict=strict) if isinstance(k, str) else k
            resolved_dict[resolved_k] = resolve_secrets(v, env=env, strict=strict)
        return resolved_dict

    if isinstance(data, list):
        return [resolve_secrets(item, env=env, strict=strict) for item in data]

    if isinstance(data, tuple):
        return tuple(resolve_secrets(item, env=env, strict=strict) for item in data)

    if isinstance(data, set):
        return {resolve_secrets(item, env=env, strict=strict) for item in data}

    return data
