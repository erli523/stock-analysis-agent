#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small helpers for normalizing Agent result metadata."""

from typing import Any, Dict


def has_simulated_source(*sources: Dict[str, Any]) -> bool:
    """Return true when any source explicitly marks its data as simulated."""
    return any(bool(source and source.get("is_simulated")) for source in sources)


def build_data_quality_fields(
    *,
    data_available: bool = True,
    data_note: str = "",
    is_simulated: bool = False,
    is_estimated: bool = False,
    partial_when_noted: bool = True,
) -> Dict[str, Any]:
    """Build the shared data-quality fields used by specialist Agent outputs.

    Priority is intentionally strict: simulated > unavailable > estimated > partial > verified.
    That keeps downstream guardrails conservative when data is synthetic or limited.
    """
    normalized_note = str(data_note or "")
    if is_simulated:
        level = "simulated"
    elif data_available is False:
        level = "unavailable"
    elif is_estimated:
        level = "estimated"
    elif partial_when_noted and normalized_note:
        level = "partial"
    else:
        level = "verified"

    return {
        "data_available": bool(data_available),
        "data_note": normalized_note,
        "data_quality_level": level,
        "is_simulated": bool(is_simulated),
    }
