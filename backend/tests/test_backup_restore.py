import os
import pytest
import tempfile
from backend.scripts.backup_restore import BackupManager


def test_backup_creation_and_checksum_integrity():
    """Verifies that database backups generate valid cryptographic checksums and metadata manifests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = BackupManager(backup_dir=temp_dir)
        manifest = manager.create_backup("test_integrity")

        assert manifest["status"] == "COMPLETED"
        assert os.path.exists(manifest["backup_file"])
        assert os.path.exists(manifest["manifest_file"])
        assert len(manifest["sha256_checksum"]) == 64

        # Verify integrity matches
        valid, msg = manager.verify_integrity(manifest["manifest_file"])
        assert valid is True
        assert "verified" in msg.lower()


def test_backup_tamper_detection():
    """Verifies that tampering with a backup file triggers an immediate checksum integrity failure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = BackupManager(backup_dir=temp_dir)
        manifest = manager.create_backup("test_tamper")

        # Intentionally tamper with the backup file by appending bytes
        with open(manifest["backup_file"], "ab") as f:
            f.write(b"CORRUPTED_BYTES_INJECTED")

        # Integrity verification must detect tampering
        valid, msg = manager.verify_integrity(manifest["manifest_file"])
        assert valid is False
        assert "failed" in msg.lower()


def test_disaster_recovery_test_restoration():
    """Verifies actual test restoration into an isolated database, measuring RTO and RPO."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = BackupManager(backup_dir=temp_dir)
        manifest = manager.create_backup("test_rto_rpo")

        restore_result = manager.test_restore_recovery(manifest["manifest_file"])
        assert restore_result["status"] == "RESTORE_VERIFIED"
        assert restore_result["rto_seconds"] is not None
        assert restore_result["rto_seconds"] < 5.0  # RTO must be well within enterprise limits
        assert restore_result["rpo_seconds"] >= 0
