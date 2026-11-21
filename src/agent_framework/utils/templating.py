"""String interpolation utility for system prompts and context injection.

Safely replaces {key} and {{key}} placeholders in prompt templates
with context data without breaking raw JSON examples or curly braces.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


# Double braces {{key}} that do NOT begin with env:
DOUBLE_BRACE_PATTERN = re.compile(r"\{\{(?!env:)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
# Single braces {key} not preceded or followed by curly braces
SINGLE_BRACE_PATTERN = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


class TemplateInterpolationError(Exception):
    """Raised when a required template variable cannot be resolved in strict mode."""

    def __init__(self, key: str, message: str | None = None) -> None:
        self.key = key
        msg = message or f"Missing template variable: '{key}'"
        super().__init__(msg)


def interpolate_prompt(
    prompt: str,
    context_data: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> str:
    """Interpolates context data into a prompt template.

    Supports both ``{key}`` and ``{{key}}`` syntax while ignoring ``{{env:...}}``
    references. In non-strict mode (default), placeholders not present in
    context_data are left untouched to prevent corrupting literal JSON examples
    or syntax blocks in system instructions.

    Args:
        prompt: System prompt containing placeholder variables.
        context_data: Key-value dictionary of values to inject.
        strict: If True, raises TemplateInterpolationError if any variable
                in the prompt is missing from context_data.

    Returns:
        The prompt string with placeholders replaced by their string values.

    Raises:
        TemplateInterpolationError: If strict=True and a placeholder is not found.
    """
    if not prompt or not isinstance(prompt, str):
        return ""

    if context_data is None:
        context_data = {}

    def _format_val(val: Any) -> str:
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)

    def _replace_double(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context_data:
            return _format_val(context_data[key])
        if strict:
            raise TemplateInterpolationError(key)
        return match.group(0)

    def _replace_single(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context_data:
            return _format_val(context_data[key])
        if strict:
            raise TemplateInterpolationError(key)
        return match.group(0)

    # First replace double braces (excluding {{env:...}}), then single braces
    result = DOUBLE_BRACE_PATTERN.sub(_replace_double, prompt)
    result = SINGLE_BRACE_PATTERN.sub(_replace_single, result)
    return result
