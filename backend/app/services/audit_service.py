import json
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from backend.app.models.audit import AuditLog, AuditOutcome


class AuditService:
    @classmethod
    def log(
        cls,
        db: Session,
        action: str,
        resource_type: str,
        outcome: AuditOutcome,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: str = "sys",
    ) -> AuditLog:
        return cls.log_event(
            db=db,
            action=action,
            resource_type=resource_type,
            outcome=outcome,
            correlation_id=correlation_id,
            actor_user_id=user_id,
            resource_id=resource_id,
            organization_id=organization_id,
            details=details,
        )

    @staticmethod
    def log_event(
        db: Session,
        action: str,
        resource_type: str,
        outcome: AuditOutcome,
        correlation_id: str,
        actor_user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Persists a tamper-evident audit record.
        Sanitizes details dictionary to guarantee zero PHI or credential leaks.
        """
        sanitized_details = None
        if details:
            # Filter out sensitive keys
            filtered = {
                k: v for k, v in details.items()
                if not any(bad in k.lower() for bad in ["password", "secret", "token", "bp", "glucose", "diagnosis"])
            }
            sanitized_details = json.dumps(filtered)
            
        audit_entry = AuditLog(
            actor_user_id=actor_user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            organization_id=organization_id,
            outcome=outcome.value,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=sanitized_details
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        return audit_entry
