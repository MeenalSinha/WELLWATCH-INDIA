import os
from typing import Dict, List, Optional, Any, Tuple
from backend.app.integrations.adapters.base import BaseIntegrationAdapter


class UaePassAdapter(BaseIntegrationAdapter):
    """National Digital Identity (UAE PASS) OpenID Connect Adapter."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("UAE_PASS", config or {})
        self.client_id = os.getenv("UAE_PASS_CLIENT_ID")
        self.client_secret = os.getenv("UAE_PASS_CLIENT_SECRET")
        self.redirect_uri = os.getenv("UAE_PASS_REDIRECT_URI")

    def get_status(self) -> str:
        if not self.client_id or not self.client_secret:
            return "NOT_CONFIGURED"
        return "CONFIGURED"

    def test_connection(self) -> Tuple[bool, str]:
        if self.get_status() == "NOT_CONFIGURED":
            return (
                False,
                "UAE PASS is NOT_CONFIGURED: Missing UAE_PASS_CLIENT_ID or UAE_PASS_CLIENT_SECRET. "
                "Requires registration with TDRA and certificate validation.",
            )
        return True, "UAE PASS OpenID Connect metadata reachable"

    def get_prerequisites(self) -> List[str]:
        return [
            "TDRA Staging / Production Access Onboarding",
            "Public Key Certificate Registered with UAE PASS Authority",
            "Approved Digital Identity Redirect URI",
        ]
