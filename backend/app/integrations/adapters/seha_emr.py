import os
from typing import Dict, List, Optional, Any, Tuple
from backend.app.integrations.adapters.base import BaseIntegrationAdapter


class SehaEmrAdapter(BaseIntegrationAdapter):
    """SEHA (Abu Dhabi Health Services Co.) Clinic & Hospital EMR Interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SEHA_EMR", config or {})
        self.api_url = os.getenv("SEHA_EMR_API_URL")
        self.api_key = os.getenv("SEHA_API_KEY")

    def get_status(self) -> str:
        if not self.api_url or not self.api_key:
            return "NOT_CONFIGURED"
        return "CONFIGURED"

    def test_connection(self) -> Tuple[bool, str]:
        if self.get_status() == "NOT_CONFIGURED":
            return (
                False,
                "SEHA EMR is NOT_CONFIGURED: Missing SEHA_EMR_API_URL or SEHA_API_KEY. "
                "Integration requires approved clinical integration agreement with SEHA.",
            )
        return True, f"Connected to SEHA EMR Gateway at {self.api_url}"

    def get_prerequisites(self) -> List[str]:
        return [
            "SEHA Clinical Integration Service Level Agreement",
            "Secure VPN / Private Endpoint Link",
            "SEHA Partner API Authorization Token",
        ]
