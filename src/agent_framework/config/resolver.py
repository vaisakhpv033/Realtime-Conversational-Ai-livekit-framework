"""Cascading metadata extractor and resolver for LiveKit voice agent jobs.

Extracts, deep-merges, and normalizes configuration across:
1. Job dispatch metadata (JobContext.job.metadata) - Base layer
2. Room metadata (JobContext.room.metadata) - Intermediate overlay
3. Remote Participant attributes / metadata - Top priority overlay

Supports participant identity targeting for multi-party rooms and raises
strict MetadataValidationError if the merged configuration fails validation.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Mapping, Optional, Union
from livekit.agents import JobContext

from agent_framework.config.schema import SessionConfig
from agent_framework.config.validator import MetadataValidationError, validate_metadata

logger = logging.getLogger("agent_framework.config.resolver")


def deep_merge(base: dict[str, Any] | None, overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively deep merges overlay dictionary into base dictionary.

    - Nested dictionaries are merged recursively.
    - Lists and primitive values in overlay replace those in base.
    - Safely handles None or non-dict arguments.
    - Neither input dictionary is modified in place; returns a new dictionary.

    Args:
        base: The lower-priority base dictionary.
        overlay: The higher-priority overlay dictionary.

    Returns:
        A new deep-merged dictionary.
    """
    if not isinstance(base, dict):
        return copy.deepcopy(overlay) if isinstance(overlay, dict) else {}
    if not isinstance(overlay, dict):
        return copy.deepcopy(base)

    merged: dict[str, Any] = copy.deepcopy(base)

    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def parse_layer_metadata(raw: Any, layer_name: str = "metadata") -> dict[str, Any]:
    """Safely parses a raw metadata layer into a Python dictionary.

    Args:
        raw: String, dictionary, or None.
        layer_name: Descriptive name of the layer for error diagnostics.

    Returns:
        A dictionary representation of the metadata.

    Raises:
        MetadataValidationError: If raw is a non-empty string that cannot be parsed as JSON,
                                 or if parsed JSON is not a dictionary.
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return copy.deepcopy(raw)

    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return {}
        try:
            parsed = json.loads(trimmed)
        except (json.JSONDecodeError, TypeError) as e:
            raise MetadataValidationError(
                f"Invalid JSON in {layer_name}: {e}",
                errors=[{"field": layer_name, "message": str(e), "type": "json_decode_error"}],
            ) from e

        if not isinstance(parsed, dict):
            raise MetadataValidationError(
                f"{layer_name} root must be a JSON object, got {type(parsed).__name__}.",
                errors=[{
                    "field": layer_name,
                    "message": f"Expected dict, got {type(parsed).__name__}",
                    "type": "type_error",
                }],
            )
        return parsed

    raise MetadataValidationError(
        f"Unexpected type for {layer_name}: expected dict or JSON str, got {type(raw).__name__}.",
        errors=[{
            "field": layer_name,
            "message": f"Expected dict or JSON str, got {type(raw).__name__}",
            "type": "type_error",
        }],
    )


def resolve_cascading_metadata(
    *layers: Union[dict[str, Any], str, None],
) -> dict[str, Any]:
    """Cascades arbitrary metadata layers in order from lowest to highest priority.

    Example:
        `resolve_cascading_metadata(job_meta, room_meta, participant_meta)`

    Args:
        *layers: Metadata sources in ascending priority order.

    Returns:
        The deep-merged result dictionary.
    """
    merged: dict[str, Any] = {}
    for idx, layer in enumerate(layers):
        layer_dict = parse_layer_metadata(layer, layer_name=f"layer[{idx}]")
        merged = deep_merge(merged, layer_dict)
    return merged


KNOWN_TOP_LEVEL_SECTIONS = {
    "version",
    "agent",
    "models",
    "instructions",
    "voice",
    "vad",
    "turn_detection",
    "session",
    "tools",
    "reporting",
    "metadata",
}


def extract_participant_metadata(
    participant: Any,
) -> dict[str, Any]:
    """Extracts and merges metadata and attributes from a LiveKit RemoteParticipant.

    Checks:
    1. participant.attributes (parsing 'metadata' JSON blob if present, mapping known top-level sections
       and routing unmapped attributes into the 'metadata' passthrough section).
    2. participant.metadata (JSON string or dict) - highest priority.

    Args:
        participant: A LiveKit participant object.

    Returns:
        Dictionary of participant-level metadata.
    """
    p_merged: dict[str, Any] = {}

    # 1. Parse participant.attributes if present
    attrs = getattr(participant, "attributes", None)
    if isinstance(attrs, dict):
        # If attributes contains a 'metadata' blob, parse it first
        if "metadata" in attrs and attrs["metadata"] is not None:
            attr_meta = parse_layer_metadata(attrs["metadata"], layer_name="participant.attributes['metadata']")
            p_merged = deep_merge(p_merged, attr_meta)

        # Merge other attributes
        for k, v in attrs.items():
            if k == "metadata":
                continue
            parsed_v = v
            if isinstance(v, str):
                trimmed_v = v.strip()
                if (trimmed_v.startswith("{") and trimmed_v.endswith("}")) or (
                    trimmed_v.startswith("[") and trimmed_v.endswith("]")
                ):
                    try:
                        parsed_v = json.loads(trimmed_v)
                    except (json.JSONDecodeError, TypeError):
                        pass

            if k in KNOWN_TOP_LEVEL_SECTIONS:
                if k in p_merged and isinstance(p_merged[k], dict) and isinstance(parsed_v, dict):
                    p_merged[k] = deep_merge(p_merged[k], parsed_v)
                else:
                    p_merged[k] = parsed_v
            else:
                # Route unmapped participant attributes into arbitrary passthrough metadata
                meta_section = p_merged.get("metadata")
                if not isinstance(meta_section, dict):
                    p_merged["metadata"] = {}
                p_merged["metadata"][k] = parsed_v

    # 2. Parse participant.metadata (JSON string) - higher priority than attributes
    meta_raw = getattr(participant, "metadata", None)
    if meta_raw:
        parsed_meta = parse_layer_metadata(meta_raw, layer_name="participant.metadata")
        p_merged = deep_merge(p_merged, parsed_meta)

    return p_merged


def extract_session_metadata(
    ctx: JobContext,
    target_participant_identity: Optional[str] = None,
    resolve_env_secrets: bool = True,
    env: Mapping[str, str] | None = None,
) -> SessionConfig:
    """Extracts, cascades, and strictly validates session metadata from a LiveKit JobContext.

    Priority Cascade (Lowest to Highest):
    1. Job Dispatch metadata (ctx.job.metadata)
    2. Room metadata (ctx.room.metadata)
    3. Remote Participant attributes / metadata

    Participant Identity Targeting:
    - If `target_participant_identity` is provided (via parameter or Job/Room session config),
      the resolver binds strictly to the participant with matching identity.
    - If not specified, the resolver binds to the first remote participant if present.

    Args:
        ctx: LiveKit JobContext for the active call.
        target_participant_identity: Optional target identity to bind with.
        resolve_env_secrets: Whether to resolve {{env:KEY}} references in metadata.
        env: Optional custom environment mapping for secret resolution.

    Returns:
        Validated SessionConfig instance.

    Raises:
        MetadataValidationError: If the merged metadata fails schema validation or required
                                 fields are missing.
    """
    merged_data: dict[str, Any] = {}

    if ctx is not None:
        # Layer 1: Job Dispatch metadata (Base)
        if ctx.job and getattr(ctx.job, "metadata", None):
            job_data = parse_layer_metadata(ctx.job.metadata, layer_name="job.metadata")
            merged_data = deep_merge(merged_data, job_data)

        # Layer 2: Room metadata (Overrides Job metadata)
        if ctx.room and getattr(ctx.room, "metadata", None):
            room_data = parse_layer_metadata(ctx.room.metadata, layer_name="room.metadata")
            merged_data = deep_merge(merged_data, room_data)

    # If target_participant_identity was explicitly passed as argument, set it in session
    if target_participant_identity:
        merged_data.setdefault("session", {})["target_participant_identity"] = target_participant_identity

    # Determine effective target participant identity
    configured_target = (
        target_participant_identity
        or merged_data.get("session", {}).get("target_participant_identity")
    )

    # Layer 3: Remote Participant attributes / metadata
    matched_participant_identity: Optional[str] = None

    if ctx is not None and ctx.room and getattr(ctx.room, "remote_participants", None):
        participants = ctx.room.remote_participants
        selected_participant = None

        if configured_target:
            # Bind specifically to targeted participant identity
            for p in participants.values():
                if getattr(p, "identity", None) == configured_target:
                    selected_participant = p
                    matched_participant_identity = configured_target
                    break
            if selected_participant is None:
                logger.warning(
                    f"Target participant '{configured_target}' not found among remote participants: "
                    f"{[getattr(p, 'identity', None) for p in participants.values()]}"
                )
        elif participants:
            # Bind to first available remote participant
            selected_participant = next(iter(participants.values()))
            matched_participant_identity = getattr(selected_participant, "identity", None)

        if selected_participant is not None:
            participant_data = extract_participant_metadata(selected_participant)
            merged_data = deep_merge(merged_data, participant_data)

    # Automatically set session.target_participant_identity if resolved
    if matched_participant_identity:
        session_section = merged_data.setdefault("session", {})
        session_section["target_participant_identity"] = matched_participant_identity

    # Strictly validate merged payload
    return validate_metadata(
        merged_data,
        resolve_env_secrets=resolve_env_secrets,
        env=env,
    )
