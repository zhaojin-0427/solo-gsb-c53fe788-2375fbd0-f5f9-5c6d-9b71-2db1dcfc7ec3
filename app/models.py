"""Pydantic 请求/响应模式。"""

from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .engine import DISTRIBUTIONS

Distribution = Literal["normal", "uniform", "triangular"]
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {"pi", "e"}


class InputSpec(BaseModel):
    """登记输入量：名义值、单位、标准不确定度、分布。"""

    name: str = Field(min_length=1, max_length=64)
    value: float
    unit: str = Field(min_length=1, max_length=64)
    standard_uncertainty: float = Field(ge=0.0)
    distribution: Distribution

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("输入名必须是标识符（字母/下划线开头，仅含字母数字下划线）")
        if v in _RESERVED:
            raise ValueError(f"{v!r} 是保留常量名，不能用作输入名")
        return v


class CorrelationPair(BaseModel):
    """输入量两两相关系数。"""

    i: str = Field(min_length=1, max_length=64)
    j: str = Field(min_length=1, max_length=64)
    r: float = Field(ge=-1.0, le=1.0)


class ModelCreate(BaseModel):
    """不可变版本化测量模型登记请求。"""

    version: str = Field(min_length=1, max_length=128)
    name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    expression: str = Field(min_length=1, max_length=4000)
    inputs: list[InputSpec] = Field(min_length=1, max_length=64)
    output_unit: str = Field(min_length=1, max_length=64)
    correlations: list[CorrelationPair] = Field(default_factory=list)


class SnapshotInput(BaseModel):
    """计算请求中的输入快照（与登记模型逐项核对）。"""

    name: str = Field(min_length=1, max_length=64)
    value: float
    unit: str = Field(min_length=1, max_length=64)
    standard_uncertainty: float = Field(ge=0.0)
    distribution: Distribution


class EvaluationCreate(BaseModel):
    """提交一次不确定度传播计算。"""

    model_version: str = Field(min_length=1, max_length=128)
    inputs: list[SnapshotInput] = Field(min_length=1, max_length=64)
    correlations: list[CorrelationPair] = Field(default_factory=list)
    n: int = Field(ge=1000, le=100_000, description="蒙特卡洛抽样次数（1000–100000）")
    seed: int = Field(ge=0, le=2**63 - 1)
    idempotency_key: str = Field(min_length=8, max_length=128)
    convergence_tolerance: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="收敛判据相对容差（均值/标准差/分位标准误）",
    )


class PrecisionTarget(BaseModel):
    """精度反求目标：带单位的非负有限 GUM 合成标准不确定度上界。"""

    standard_uncertainty: float = Field(description="目标 GUM 合成标准不确定度（必须为非负有限值）")
    unit: str = Field(min_length=1, max_length=64, description="目标值单位（须与模型输出量纲一致）")

    @field_validator("standard_uncertainty")
    @classmethod
    def _non_negative_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError("目标标准不确定度必须是非负有限数（不接受 NaN/Inf/负值）")
        return v


class PrecisionBudgetCreate(BaseModel):
    """测量精度反求请求：只变动一个已登记输入的标准不确定度。"""

    model_version: str = Field(min_length=1, max_length=128)
    input_name: str = Field(min_length=1, max_length=64)
    target: PrecisionTarget


__all__ = [
    "Distribution",
    "InputSpec",
    "CorrelationPair",
    "ModelCreate",
    "SnapshotInput",
    "EvaluationCreate",
    "PrecisionTarget",
    "PrecisionBudgetCreate",
    "DISTRIBUTIONS",
]
