"""Basic reusable types for the Python port."""

from dataclasses import dataclass


@dataclass
class NodeStats:
    visits: int = 0
    value: float = 0.0
