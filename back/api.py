"""兼容的 ASGI 入口；正式实现位于接口层。"""

from back.interfaces.http.app import app


__all__ = ["app"]
