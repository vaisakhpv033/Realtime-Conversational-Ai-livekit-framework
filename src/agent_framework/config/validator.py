"""Strict metadata validator for LiveKit Voice Agent Framework.

Enforces zero backward compatibility, zero guessing, and fail-fast validation.
Raises structured MetadataValidationError on any missing required key,
extra top-level field, or invalid hyperparameter.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Union
from pydantic import ValidationError

from agent_framework.config.schema import SessionConfig
from agent_framework.utils.secrets import SecretResolutionError, resolve_secrets


class MetadataValidationError(Exception):
    """Raised when metadata fails validation against the SessionConfig schema."""

    def __init__(
        self,
        message: str,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.message = message
        self.errors = errors or []
        super().__init__(message)


def validate_metadata(
    data: Union[dict[str, Any], str, SessionConfig],
    resolve_env_secrets: bool = False,
    env: Mapping[str, str] | None = None,
) -> SessionConfig:
    """Strictly validates input metadata against the SessionConfig schema.

    Args:
        data: Raw metadata as a dictionary, a JSON string, or an existing SessionConfig.
        resolve_env_secrets: If True, resolves {{env:KEY}} references prior to validation.
        env: Optional custom environment mapping for secret resolution.

    Returns:
        A validated, typed SessionConfig instance.

    Raises:
        MetadataValidationError: If the payload is malformed, missing required keys,
                                 contains disallowed extra keys, or fails constraints.
    """
    dict_data: dict[str, Any]

    if isinstance(data, SessionConfig):
        if not resolve_env_secrets:
            return data
        dict_data = data.model_dump()
    elif isinstance(data, str):
        trimmed = data.strip()
        if not trimmed:
            raise MetadataValidationError(
                "Metadata payload cannot be empty string.",
                errors=[{"field": "$root", "message": "Empty string provided", "type": "empty_input"}],
            )
        try:
            parsed = json.loads(trimmed)
        except (json.JSONDecodeError, TypeError) as e:
            raise MetadataValidationError(
                f"Invalid JSON metadata payload: {e}",
                errors=[{"field": "$root", "message": str(e), "type": "json_decode_error"}],
            ) from e

        if not isinstance(parsed, dict):
            raise MetadataValidationError(
                f"Metadata root must be a JSON object, got {type(parsed).__name__}.",
                errors=[{
                    "field": "$root",
                    "message": f"Expected dict, got {type(parsed).__name__}",
                    "type": "type_error",
                }],
            )
        dict_data = parsed

    elif isinstance(data, dict):
        dict_data = dict(data)
    else:
        raise MetadataValidationError(
            f"Expected metadata to be a dict, JSON string, or SessionConfig; got {type(data).__name__}.",
            errors=[{
                "field": "$root",
                "message": f"Invalid type {type(data).__name__}",
                "type": "type_error",
            }],
        )

    # Optionally resolve {{env:KEY}} secret references before validation
    if resolve_env_secrets:
        try:
            dict_data = resolve_secrets(dict_data, env=env, strict=True)
        except SecretResolutionError as e:
            raise MetadataValidationError(
                f"Failed to resolve environment secret in metadata: {e}",
                errors=[{
                    "field": "$root",
                    "message": str(e),
                    "type": "secret_resolution_error",
                }],
            ) from e

    try:
        return SessionConfig.model_validate(dict_data)
    except ValidationError as e:
        error_details: list[dict[str, Any]] = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            msg = err.get("msg", "")
            err_type = err.get("type", "")
            input_val = err.get("input")
            error_details.append({
                "field": loc or "$root",
                "message": msg,
                "type": err_type,
                "input": input_val,
            })

        summary = (
            f"Metadata validation failed with {len(error_details)} error(s):\n"
            + "\n".join(f"  - {err['field']}: {err['message']}" for err in error_details)
        )
        raise MetadataValidationError(summary, errors=error_details) from e
