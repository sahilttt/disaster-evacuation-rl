"""
Shared helper functions.

Utility functions used across multiple modules.
"""

import os
import json
from typing import Any, Dict, List


def ensure_directory(path: str) -> None:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary as a JSON file.

    Args:
        data: Dictionary to serialise.
        filepath: Destination file path.
    """
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: List[Dict[str, Any]], filepath: str) -> None:
    """Persist a training-episode history list as JSON.

    Args:
        history: List of per-episode metric dicts (from Trainer).
        filepath: Destination file path.
    """
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_history(filepath: str) -> List[Dict[str, Any]]:
    """Load a previously saved training history.

    Args:
        filepath: Path to the JSON history file.

    Returns:
        List of per-episode metric dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

