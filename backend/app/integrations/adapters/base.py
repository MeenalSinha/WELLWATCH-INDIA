from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class BaseIntegrationAdapter(ABC):
    """Abstract Base Class for external healthcare platform adapters."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def get_status(self) -> str:
        """Returns genuine operational status: NOT_CONFIGURED, CONFIGURED, CONNECTED, or BLOCKED."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Performs a live connectivity and authentication handshake."""
        pass

    @abstractmethod
    def get_prerequisites(self) -> List[str]:
        """Returns list of formal regulatory or technical prerequisites required for live connection."""
        pass
