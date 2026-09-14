"""可跨接入渠道使用的业务错误；HTTP 状态映射由网关负责。"""


class ApplicationError(Exception):
    """对外可解释的用例错误。"""


class InvalidRequest(ApplicationError):
    pass


class NotFound(ApplicationError):
    pass


class Conflict(ApplicationError):
    pass


class DependencyUnavailable(ApplicationError):
    pass


class ProcessingFailed(ApplicationError):
    pass
