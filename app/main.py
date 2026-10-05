"""FastAPI 入口：路由、错误映射、启动初始化。"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__, services, storage
from .errors import DomainError
from .models import EvaluationCreate, ModelCreate, PrecisionBudgetCreate


def _sanitize(obj):
    """把错误结构中的 NaN/Inf 及异常对象替换为可 JSON 序列化形式。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.init_db()
    yield


app = FastAPI(
    title="实验测量不确定度传播 API",
    description=(
        "登记带单位的输入量、标准不确定度、分布、相关系数与不可变版本化测量模型；"
        "提交后执行 GUM 一阶线性化与固定种子蒙特卡洛传播。\n\n"
        "表达式仅允许白名单算子与函数；请求按 模型版本+输入快照+N+种子+幂等键 绑定。"
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(DomainError)
async def _domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
    # 所有可预期的拒绝：不保存任何结果，返回结构化错误
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # 默认处理器会把被拒绝的 NaN/Inf 原样放进错误明细，导致响应本身不是合法 JSON；
    # 这里统一净化非有限浮点后再返回 422。
    return JSONResponse(status_code=422, content=_sanitize(exc.errors()))


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"service": "uncertainty-propagation-api", "version": __version__, "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

@app.post("/api/v1/models", status_code=201, tags=["models"])
def create_model(payload: ModelCreate) -> dict:
    """登记不可变版本化测量模型（重复版本 → 409）。"""
    return services.register_model(payload)


@app.get("/api/v1/models/{version}", tags=["models"])
def get_model(version: str) -> dict:
    return storage.get_model(version)


@app.get("/api/v1/models", tags=["models"])
def get_models(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    rows = storage.list_models(limit=limit, offset=offset)
    return {"items": rows, "limit": limit, "offset": offset, "count": len(rows)}


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

@app.post("/api/v1/evaluations", status_code=201, tags=["evaluations"])
def create_evaluation(payload: EvaluationCreate) -> JSONResponse:
    """提交计算。重复请求（相同幂等键/请求指纹）只返回同一结果（200 + cached=true）。

    非法相关矩阵、维度不兼容或蒙特卡洛不收敛一律 422 拒绝，且不保存任何结果。
    """
    result = services.submit_evaluation(payload)
    cached = result.get("cached", False)
    # 去重命中返回 200；首次计算返回 201
    return JSONResponse(status_code=200 if cached else 201, content=result)


@app.get("/api/v1/evaluations/{evaluation_id}", tags=["evaluations"])
def get_evaluation(evaluation_id: str) -> dict:
    return storage.get_evaluation(evaluation_id)


@app.get("/api/v1/evaluations", tags=["evaluations"])
def get_evaluations(
    model_version: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    rows = storage.list_evaluations(model_version=model_version, limit=limit, offset=offset)
    return {"items": rows, "limit": limit, "offset": offset, "count": len(rows)}


@app.get("/api/v1/idempotency/{key}", tags=["evaluations"])
def get_by_idempotency_key(key: str) -> dict:
    from .errors import NotFoundError

    row = storage.find_evaluation_by_idempotency_key(key)
    if row is None:
        raise NotFoundError(f"幂等键 {key!r} 没有对应的计算结果")
    return row


# ---------------------------------------------------------------------------
# 测量精度反求
# ---------------------------------------------------------------------------

@app.post("/api/v1/precision-budgets", tags=["precision-budgets"])
def create_precision_budget(payload: PrecisionBudgetCreate) -> dict:
    """反求单个输入标准不确定度的可行区间（只读，不改模型、不写计算记录）。

    沿用模型登记的单位换算、相关矩阵与名义点灵敏度；协方差（含负相关）
    全部纳入合成方差二次不等式，响应给出输入声明单位下的端点及开闭性、
    当前登记值是否达标。
    """
    return services.solve_precision_budget(payload)
