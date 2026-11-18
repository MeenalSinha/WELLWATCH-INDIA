import os
from typing import Dict, List, Optional, Any, Tuple
from backend.app.integrations.adapters.base import BaseIntegrationAdapter


class LabNetworkAdapter(BaseIntegrationAdapter):
    """National Reference Laboratory & Abu Dhabi Diagnostic Network Adapter."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("LAB_NETWORK", config or {})
        self.api_url = os.getenv("LAB_NETWORK_API_URL")
        self.api_token = os.getenv("LAB_NETWORK_TOKEN")

    def get_status(self) -> str:
        if not self.api_url or not self.api_token:
            return "NOT_CONFIGURED"
        return "CONFIGURED"

    def test_connection(self) -> Tuple[bool, str]:
        if self.get_status() == "NOT_CONFIGURED":
            return (
                False,
                "Abu Dhabi Laboratory Network is NOT_CONFIGURED: Missing LAB_NETWORK_API_URL or LAB_NETWORK_TOKEN.",
            )
        return True, f"Connected to Laboratory Gateway at {self.api_url}"

    def get_prerequisites(self) -> List[str]:
        return [
            "Clinical Laboratory Data Exchange Authorization",
            "Diagnostic Result Encryption Keys",
        ]
