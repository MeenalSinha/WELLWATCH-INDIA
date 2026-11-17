from starlette.responses import Response
from backend.app.middleware.security_headers import SecurityHeadersMiddleware
from backend.app.middleware.correlation import CorrelationIdMiddleware


def test_security_headers_middleware_direct():
    response = Response("OK")
    response = SecurityHeadersMiddleware.apply_headers(response)

    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "Strict-Transport-Security" in response.headers
    assert "Content-Security-Policy" in response.headers
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_correlation_id_middleware_direct():
    response = Response("OK")
    correlation_id = "test-correlation-uuid-12345"
    response = CorrelationIdMiddleware.inject_correlation_id(response, correlation_id)

    assert response.headers.get("X-Correlation-ID") == correlation_id
