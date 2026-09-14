"""ASGI 组装入口：路由、生命周期和统一错误映射。"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from back.bootstrap import initialize_database, initialize_search, preload_models
from back.domain.errors import ApplicationError, Conflict, DependencyUnavailable, InvalidRequest, NotFound, ProcessingFailed
from back.interfaces.http.admin import router as admin_router
from back.interfaces.http.chat import router as chat_router
from back.interfaces.http.public import router as public_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_checks = {}
    for name, initialize in (("database", initialize_database), ("search", initialize_search), ("models", preload_models)):
        try:
            await run_in_threadpool(initialize)
            app.state.startup_checks[name] = "ok"
        except Exception:
            logger.exception("组件初始化失败：%s；其他接口继续提供服务", name)
            app.state.startup_checks[name] = "unavailable"
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="AI Customer Service Agent", version="0.2.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False, allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Admin-Key", "X-Tenant-Token", "Authorization"],
    )

    @application.exception_handler(ApplicationError)
    async def application_error(_request: Request, error: ApplicationError):
        status = {InvalidRequest: 400, NotFound: 404, Conflict: 409, DependencyUnavailable: 503, ProcessingFailed: 422}
        return JSONResponse(status_code=status.get(type(error), 500), content={"detail": str(error)})

    @application.get("/health")
    def health_check():
        checks = getattr(application.state, "startup_checks", {})
        return {"status": "degraded" if "unavailable" in checks.values() else "ok", "version": "0.2.0"}

    application.include_router(public_router)
    application.include_router(admin_router)
    application.include_router(chat_router)
    return application


app = create_app()
