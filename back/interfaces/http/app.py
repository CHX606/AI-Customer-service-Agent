"""ASGI 组装入口：路由、生命周期和统一错误映射。"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from back.bootstrap import get_handoff_service, initialize_database, initialize_search, preload_models
from back.core.features import image_features_enabled
from back.domain.errors import ApplicationError, Conflict, DependencyUnavailable, InvalidRequest, NotFound, ProcessingFailed
from back.interfaces.http.admin import router as admin_router
from back.interfaces.http.chat import router as chat_router
from back.interfaces.http.handoff import router as handoff_router
from back.interfaces.http.public import router as public_router

logger = logging.getLogger(__name__)


def _configure_application_logging() -> None:
    """为应用日志提供默认输出，同时复用部署方或测试已配置的 handler。"""
    application_logger = logging.getLogger("back")
    if application_logger.level == logging.NOTSET:
        application_logger.setLevel(logging.INFO)

    fallback = next(
        (handler for handler in application_logger.handlers
         if handler.get_name() == "customer_service_default"),
        None,
    )
    propagate = (
        getattr(fallback, "_original_propagate", application_logger.propagate)
        if fallback is not None else application_logger.propagate
    )
    handlers = [handler for handler in application_logger.handlers if handler is not fallback]
    if propagate:
        handlers.extend(logging.getLogger().handlers)
    has_external_output = any(
        handler.level <= logging.INFO and not isinstance(handler, logging.NullHandler)
        for handler in handlers
    )
    if has_external_output:
        if fallback is not None:
            application_logger.removeHandler(fallback)
            fallback.close()
            application_logger.propagate = propagate
        return
    if fallback is None:
        fallback = logging.StreamHandler()
        fallback.set_name("customer_service_default")
        fallback.setLevel(logging.INFO)
        fallback.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        fallback._original_propagate = application_logger.propagate
        application_logger.addHandler(fallback)
        application_logger.propagate = False


HANDOFF_WORKER_SHUTDOWN_SECONDS = 5


async def _deliver_handoff_notifications(stop: asyncio.Event, executor: ThreadPoolExecutor) -> None:
    while not stop.is_set():
        try:
            await asyncio.get_running_loop().run_in_executor(
                executor, partial(get_handoff_service().dispatch_pending, limit=1))
        except Exception as error:
            # Delivery failures must never stop the chat service or reveal SMTP configuration.
            logger.warning("人工通知队列检查失败：%s", type(error).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_application_logging()
    app.state.startup_checks = {}
    for name, initialize in (("database", initialize_database), ("search", initialize_search), ("models", preload_models)):
        try:
            await run_in_threadpool(initialize)
            app.state.startup_checks[name] = "ok"
        except Exception:
            logger.exception("组件初始化失败：%s；其他接口继续提供服务", name)
            app.state.startup_checks[name] = "unavailable"
    stop = asyncio.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="handoff-mail")
    worker = asyncio.create_task(_deliver_handoff_notifications(stop, executor))
    try:
        yield
    finally:
        stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=HANDOFF_WORKER_SHUTDOWN_SECONDS)
        except TimeoutError:
            # Persisted claims survive shutdown; uncertain delivery requires operator review.
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        except Exception as error:
            logger.warning("人工通知任务退出失败：%s", type(error).__name__)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


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
        return {"status": "degraded" if "unavailable" in checks.values() else "ok", "version": "0.2.0",
                "features": {"image_chat": image_features_enabled()}}

    application.include_router(public_router)
    application.include_router(admin_router)
    application.include_router(chat_router)
    application.include_router(handoff_router)
    return application


app = create_app()
