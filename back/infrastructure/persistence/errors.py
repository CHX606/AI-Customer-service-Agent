"""存储故障转换，不将驱动错误或 SQL 内容暴露给上层。"""
from functools import wraps
import logging
import sqlite3
from back.domain.errors import DependencyUnavailable

logger = logging.getLogger(__name__)


def storage_operation(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (sqlite3.Error, OSError) as exc:
            logger.exception("存储操作失败：%s", function.__qualname__)
            raise DependencyUnavailable("数据存储暂不可用，请稍后重试。") from exc
    return invoke
