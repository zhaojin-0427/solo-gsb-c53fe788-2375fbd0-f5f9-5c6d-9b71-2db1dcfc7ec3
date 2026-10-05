"""单位换算与维度推导（基于 Pint）。

计算管线统一在 *SI 基本单位量纲* 的数值空间中进行 NumPy 运算：

  x_base = b_i * x_declared      （b_i 为声明单位 -> 基本单位的换算因子）
  y_out  = s   * y_base          （s 为输出基本单位 -> 输出单位的换算因子）

这样蒙特卡洛数组完全不带单位对象，速度快；维度正确性由 Pint 单独推导一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pint

from .errors import ModelError, RequestError

_UREG: pint.UnitRegistry | None = None


def get_registry() -> pint.UnitRegistry:
    global _UREG
    if _UREG is None:
        reg = pint.UnitRegistry()
        # 角度按普通量处理：模型中 sin/cos 等要求显式无量纲
        _UREG = reg
    return _UREG


def parse_unit(spec: str) -> pint.Unit:
    """把字符串解析为 Pint 单位，失败转成 ModelError。"""
    reg = get_registry()
    try:
        return reg.parse_units(spec)
    except pint.errors.UndefinedUnitError as exc:
        raise ModelError(f"未知单位: {spec!r} ({exc.unit_names})") from exc
    except Exception as exc:  # pragma: no cover - 防御式
        raise ModelError(f"无法解析单位 {spec!r}: {exc}") from exc


def to_declared_magnitude(value: float, unit_spec: str, declared: pint.Unit) -> float:
    """把请求中 ``value <unit_spec>`` 的数值换算成模型声明单位下的数值。"""
    reg = get_registry()
    src = parse_unit(unit_spec)
    try:
        q = reg.Quantity(float(value), src).to(declared)
    except pint.errors.DimensionalityError as exc:
        raise RequestError(
            f"输入单位 {unit_spec!r} 与模型声明单位 {declared:~} 量纲不兼容"
        ) from exc
    return float(q.magnitude)


def base_factor(unit: pint.Unit) -> float:
    """1 个声明单位 = 多少个 SI 基本单位下的数值。"""
    reg = get_registry()
    return float(reg.Quantity(1.0, unit).to_base_units().magnitude)


# ---------------------------------------------------------------------------
# 白名单函数的 Pint 版本（维度推导用）
# ---------------------------------------------------------------------------

def _pint_functions(reg: pint.UnitRegistry) -> dict[str, object]:
    def dimless_maker(fn):
        def wrapped(q):
            if not isinstance(q, pint.Quantity):
                q = reg.Quantity(q, "")
            if not q.dimensionless:
                raise RequestError(f"函数 {fn.__name__} 的参数必须无量纲，得到 {q.dimensionality}")
            return reg.Quantity(fn(q.m_as("")), "")

        wrapped.__name__ = fn.__name__
        return wrapped

    funcs = {
        name: dimless_maker(fn)
        for name, fn in (
            ("exp", np.exp),
            ("log", np.log),
            ("log10", np.log10),
            ("sin", np.sin),
            ("cos", np.cos),
            ("tan", np.tan),
            ("asin", np.arcsin),
            ("acos", np.arccos),
            ("atan", np.arctan),
            ("sinh", np.sinh),
            ("cosh", np.cosh),
            ("tanh", np.tanh),
        )
    }

    def sqrt(q):
        if not isinstance(q, pint.Quantity):
            q = reg.Quantity(q, "")
        return q ** 0.5

    def abs_(q):
        if not isinstance(q, pint.Quantity):
            q = reg.Quantity(q, "")
        return abs(q)

    def atan2(a, b):
        qa = a if isinstance(a, pint.Quantity) else reg.Quantity(a, "")
        qb = b if isinstance(b, pint.Quantity) else reg.Quantity(b, "")
        try:
            qb = qb.to(qa.units)
        except pint.errors.DimensionalityError as exc:
            raise RequestError("atan2 的两个参数量纲必须相同") from exc
        return reg.Quantity(np.arctan2(qa.magnitude, qb.magnitude), "")

    funcs.update(sqrt=sqrt, abs=abs_, atan2=atan2)
    return funcs


@dataclass(frozen=True)
class UnitContext:
    """一个模型版本对应的单位换算上下文（不可变）。"""

    declared_units: Mapping[str, pint.Unit]
    base_factors: Mapping[str, float]
    output_unit: pint.Unit
    output_scale: float  # y_out = y_base * output_scale

    def nominal_to_base(self, declared_magnitudes: Mapping[str, float]) -> dict[str, float]:
        return {name: self.base_factors[name] * declared_magnitudes[name] for name in self.declared_units}

    def std_to_base(self, declared_std: Mapping[str, float]) -> dict[str, float]:
        return {name: self.base_factors[name] * declared_std[name] for name in self.declared_units}

    def from_base(self, y_base: np.ndarray | float) -> np.ndarray:
        return np.asarray(y_base, dtype=float) * self.output_scale

    def sensitivity(self, name: str, dbase: float) -> float:
        """基本单位空间中的导数 -> 输出单位 / 该输入声明单位。"""
        return self.output_scale * dbase * self.base_factors[name]


def build_unit_context(
    compiled,
    declared_units: Mapping[str, pint.Unit],
    output_unit_spec: str,
) -> UnitContext:
    """用 Pint 推导输出维度，并校验输出单位与之一致。

    同时完成表达式整体的维度兼容性检查（如 ``meter + second`` 在此处被拒绝）。
    """
    from .expressions import CONSTANTS

    reg = get_registry()
    env = {name: reg.Quantity(1.0, unit) for name, unit in declared_units.items()}
    env.update({name: reg.Quantity(value, "") for name, value in CONSTANTS.items()})
    funcs = _pint_functions(reg)

    try:
        sample = compiled.evaluate(env, funcs)
    except RequestError:
        raise
    except pint.errors.DimensionalityError as exc:
        raise RequestError(
            f"模型维度不兼容: 无法对 {exc.units1!r} 与 {exc.units2!r} 执行 {exc.extra_msg}"
        ) from exc
    except Exception as exc:
        raise RequestError(f"模型维度推导失败: {exc}") from exc

    if not isinstance(sample, pint.Quantity):
        sample = reg.Quantity(sample, "")

    output_unit = parse_unit(output_unit_spec)
    try:
        reg.Quantity(1.0, sample.units).to(output_unit)
    except pint.errors.DimensionalityError as exc:
        raise ModelError(
            f"声明输出单位 {output_unit_spec!r} 与模型输出维度 "
            f"{format_dimensionality(sample.dimensionality)} 不兼容"
        ) from exc

    base_q = sample.to_base_units()
    output_scale = float(reg.Quantity(1.0, base_q.units).to(output_unit).magnitude)

    return UnitContext(
        declared_units=dict(declared_units),
        base_factors={name: base_factor(u) for name, u in declared_units.items()},
        output_unit=output_unit,
        output_scale=output_scale,
    )


def format_dimensionality(dim: pint.util.UnitsContainer) -> str:
    """把 Pint 量纲容器规范化为字符串，如 ``[length]·[time]^-2``。"""
    items = sorted(dim.items())
    if not items:
        return "dimensionless"
    parts = []
    for name, exponent in items:
        parts.append(name if exponent == 1 else f"{name}^{int(exponent) if float(exponent).is_integer() else exponent}")
    return "·".join(parts)
