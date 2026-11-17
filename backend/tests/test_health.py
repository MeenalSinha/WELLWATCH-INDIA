from backend.app.api.v1.endpoints.health import get_health, get_readiness


def test_health_endpoint():
    res = get_health()
    assert res.status == "healthy"
    assert res.version == "1.0.0"
    assert "Abu Dhabi" in res.jurisdiction
    assert res.database == "connected"


def test_readiness_endpoint():
    res = get_readiness()
    assert res.status == "ready"
    assert res.database_ready is True
    assert res.checks["database"] is True
    assert res.checks["configuration"] is True
