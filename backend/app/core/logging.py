import logging
import sys
from typing import Any, Dict


class PrivacyPreservingFormatter(logging.Formatter):
    """
    Structured logging formatter that enforces privacy standards.
    Redacts known sensitive keys such as passwords, tokens, full national IDs,
    and clinical observation bodies.
    """
    SENSITIVE_KEYS = {
        "password", "secret", "token", "access_token", "refresh_token",
        "authorization", "national_id", "emirates_id", "mfa_secret",
        "systolic_bp", "fasting_glucose", "diagnosis", "notes"
    }

    def format(self, record: logging.LogRecord) -> str:
        # Standard formatted output
        orig_msg = record.getMessage()
        # Redact any obvious keyword leaks
        for key in self.SENSITIVE_KEYS:
            if key in orig_msg.lower() and "=" in orig_msg:
                orig_msg = f"[REDACTED_SENSITIVE_FIELD: {key}]"
                break
        
        record.msg = orig_msg
        return super().format(record)


def setup_logging():
    logger = logging.getLogger("wellwatch_prevent")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = PrivacyPreservingFormatter(
            fmt="%(asctime)s [%(levelname)s] [cid=%(correlation_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


logger = setup_logging()
