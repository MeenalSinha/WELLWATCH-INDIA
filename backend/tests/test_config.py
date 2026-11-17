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


def test_production_fails_safely_on_default_secret():
    prod_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="wellwatch-prevent-secure-development-secret-key-32chars-min",
        DATABASE_URL="postgresql://user:pass@localhost:5432/db"
    )
    with pytest.raises(ValueError, match="cannot use the development default"):
        Settings.validate_production_settings(prod_settings)


def test_production_fails_safely_on_sqlite():
    prod_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="super-secret-production-random-key-64-bytes-length-xyz",
        DATABASE_URL="sqlite:///./data/prod.db"
    )
    with pytest.raises(ValueError, match="must point to a production-grade database"):
        Settings.validate_production_settings(prod_settings)
