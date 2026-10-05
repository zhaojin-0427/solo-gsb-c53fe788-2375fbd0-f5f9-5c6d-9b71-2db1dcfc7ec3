"""领域错误：所有可预期的拒绝都转换为 HTTP 422（或 404/409）。"""

from __future__ import annotations


class DomainError(Exception):
    """业务校验失败基类。"""

    status_code: int = 422


class ExpressionError(DomainError):
    """模型表达式非法（语法错误 / 非白名单结构）。"""


class ModelError(DomainError):
    """模型登记错误（重复版本、单位/维度非法、表达式不一致等）。"""


class RequestError(DomainError):
    """计算请求错误（输入快照与模型不符、参数越界等）。"""


class CorrelationError(DomainError):
    """相关系数矩阵非法（不对称、越界、非半正定）。"""


class PropagationError(DomainError):
    """传播计算本身失败（求值异常、非有限值）。"""


class ConvergenceError(DomainError):
    """蒙特卡洛未通过收敛判据。"""


class NotFoundError(DomainError):
    """资源不存在。"""

    status_code = 404


class ConflictError(DomainError):
    """幂等键冲突 / 不可变版本冲突。"""

    status_code = 409
