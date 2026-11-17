import pytest
from backend.app.core.config import Settings


def test_settings_load_successfully():
    settings = Settings()
    assert settings.PROJECT_NAME.startswith("WELLWATCH PREVENT")
    assert settings.API_V1_STR == "/api/v1"
    assert len(settings.SECRET_KEY) >= 32
    assert settings.MAX_LOGIN_ATTEMPTS == 5
    assert settings.LOCKOUT_DURATION_MINUTES == 15


def test_short_secret_key_rejected():
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(SECRET_KEY="short-key")
