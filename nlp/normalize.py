"""Shared text normalisation used across all NLP modules."""

import re


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).lower().strip())
