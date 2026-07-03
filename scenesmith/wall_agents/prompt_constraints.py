"""Helpers for extracting deterministic wall-agent prompt constraints."""

import re

_MEDIA_DISPLAY_PATTERN = re.compile(
    r"\b(tv|television|monitor|screen|display)\b", re.IGNORECASE
)
_WALL_PLACEMENT_PATTERN = re.compile(
    r"\b(wall|wall-mounted|mounted|hung|hanging)\b", re.IGNORECASE
)
_MEDIA_FURNITURE_PATTERN = re.compile(
    r"\b(tv stand|television stand|media console|media cabinet|entertainment center)\b",
    re.IGNORECASE,
)


def build_required_wall_object_constraints(room_description: str) -> str:
    """Extract explicit wall-object obligations from the room prompt."""
    normalized = " ".join(room_description.split())
    lower_text = normalized.lower()
    requirements: list[str] = []

    has_media_display = bool(_MEDIA_DISPLAY_PATTERN.search(normalized))
    has_wall_hint = bool(_WALL_PLACEMENT_PATTERN.search(normalized))
    has_media_furniture = bool(_MEDIA_FURNITURE_PATTERN.search(normalized))

    # 7.3 fix: force prompt-mentioned TVs/displays into the wall stage when the
    # prompt ties them to a wall or to media furniture like a TV stand.
    if has_media_display and (has_wall_hint or has_media_furniture):
        relation_bits: list[str] = []
        if "opposite wall" in lower_text:
            relation_bits.append("use the opposite wall called out in the prompt")
        if has_media_furniture:
            relation_bits.append(
                "treat it as the focal display associated with the nearby TV stand/media console"
            )
        relation_suffix = f" ({'; '.join(relation_bits)})" if relation_bits else ""
        requirements.append(
            "- REQUIRED media display: place a wall-mounted television/display"
            f"{relation_suffix}. Do not defer it to manipulands."
        )

    if not requirements:
        return (
            "- No explicit wall-object obligations were extracted from the prompt. "
            "Decorate walls contextually."
        )

    return "\n".join(requirements)
