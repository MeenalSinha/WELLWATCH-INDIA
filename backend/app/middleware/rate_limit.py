import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.app.core.config import settings


class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate-limiting middleware to protect against brute-force
    and denial-of-service attempts.
    """
    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Exempt health and readiness checks
        if request.url.path in ["/api/v1/health", "/api/v1/ready", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean timestamps older than window
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < self.window_seconds]
        
        if len(self.clients[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please slow down and try again shortly."
                    }
                },
                headers={"Retry-After": str(self.window_seconds)}
            )
            
        self.clients[client_ip].append(now)
        return await call_next(request)
