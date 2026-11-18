import json
import os
from typing import Dict, List, Optional, Any

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "configs",
    "terminology_mappings.json",
)


class TerminologyMapper:
    """Governed Clinical Terminology Mapping Framework for LOINC, SNOMED CT, ICD-10, and UCUM."""

    _mappings: List[Dict[str, Any]] = []
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls):
        if not cls._loaded:
            normalized_path = os.path.abspath(CONFIG_PATH)
            if os.path.exists(normalized_path):
                with open(normalized_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls._mappings = data.get("mappings", [])
            cls._loaded = True

    @classmethod
    def get_all_mappings(cls) -> List[Dict[str, Any]]:
        cls._ensure_loaded()
        return cls._mappings

    @classmethod
    def lookup_internal(cls, internal_code: str) -> Optional[Dict[str, Any]]:
        """Resolves an internal clinical code to official standard terminology (LOINC/SNOMED/ICD-10)."""
        cls._ensure_loaded()
        for m in cls._mappings:
            if m.get("internal_code") == internal_code and m.get("status") == "APPROVED":
                return m
        return None

    @classmethod
    def reverse_lookup(cls, target_system: str, target_code: str) -> Optional[Dict[str, Any]]:
        """Resolves an external standard code back to canonical internal representation."""
        cls._ensure_loaded()
        for m in cls._mappings:
            if (
                m.get("target_system") == target_system
                and m.get("target_code") == target_code
                and m.get("status") == "APPROVED"
            ):
                return m
        return None
