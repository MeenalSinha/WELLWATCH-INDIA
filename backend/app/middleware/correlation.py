import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every HTTP request carries an identifiable X-Correlation-ID."""
    @classmethod
    def inject_correlation_id(cls, response: Response, correlation_id: str) -> Response:
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
            
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        return self.inject_correlation_id(response, correlation_id)
