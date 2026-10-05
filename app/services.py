"""业务编排：模型登记、快照校验、GUM+MC 计算管线。"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import uuid
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pint

from . import storage
from .engine import (
    _EPS,
    _EPS13,
    build_correlation_matrix,
    numerical_gradient_with_noise,
    run_gum,
    run_monte_carlo,
)
from .errors import ConflictError, ModelError, PropagationError, RequestError
from .expressions import compile_expression
from .models import EvaluationCreate, ModelCreate, PrecisionBudgetCreate
from .units import build_unit_context, format_dimensionality, get_registry, parse_unit, to_declared_magnitude

#: 模型版本 -> (CompiledExpression, UnitContext)；登记时构建，进程内复用
_CONTEXT_CACHE: dict[str, tuple[Any, Any]] = {}

_VALUE_MATCH_REL = 1e-9
_VALUE_MATCH_ABS = 1e-12


# ---------------------------------------------------------------------------
# 规范化工具
# ---------------------------------------------------------------------------

def canonical_expression(source: str) -> str:
    """用 AST 重新输出表达式：等价的空格/括号写法规范化一致。"""
    try:
        return ast.unparse(ast.parse(source, mode="eval"))
    except SyntaxError as exc:
        raise ModelError(f"表达式语法错误: {exc.msg}") from exc


def _pairs(pairs: list[Any]) -> list[list[float]]:
    out = [[p.i, p.j, round(float(p.r), 15)] for p in pairs]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _input_names(inputs: list[Any]) -> tuple[str, ...]:
    names = [i.name for i in inputs]
    if len(set(names)) != len(names):
        raise ModelError("输入量名称必须唯一")
    return tuple(names)


def _validate_pair_names(pairs: list[Any], names: tuple[str, ...], where: str) -> None:
    name_set = set(names)
    seen: set[frozenset[str]] = set()
    for p in pairs:
        if p.i not in name_set or p.j not in name_set:
            raise RequestError(f"{where}中相关系数引用了不存在的输入量: {p.i!r}, {p.j!r}")
        if p.i == p.j:
            raise RequestError(f"{where}中输入量 {p.i!r} 不能与自身相关")
        key = frozenset((p.i, p.j))
        if key in seen:
            raise RequestError(f"{where}中相关系数 ({p.i!r}, {p.j!r}) 重复指定")
        seen.add(key)


# ---------------------------------------------------------------------------
# 模型登记
# ---------------------------------------------------------------------------

def register_model(req: ModelCreate) -> dict[str, Any]:
    names = _input_names(req.inputs)

    # 1. 表达式白名单校验
    compiled = compile_expression(req.expression, names)

    # 2. 单位解析 + 维度推导（拒绝量纲不一致的表达式与输出单位）
    declared_units = {}
    for spec in req.inputs:
        declared_units[spec.name] = parse_unit(spec.unit)
    _validate_pair_names(req.correlations, names, "模型")
    ctx = build_unit_context(compiled, declared_units, req.output_unit)

    # 3. 登记时刻即可校验相关矩阵半正定（使用登记的标准差）
    std_decl = np.array([i.standard_uncertainty for i in req.inputs], dtype=float)
    zero_std = {names[k] for k in np.where(std_decl == 0.0)[0]}
    corr_pairs = [(p.i, p.j, p.r) for p in req.correlations]
    build_correlation_matrix(names, corr_pairs, zero_std)

    reg = get_registry()
    expr_canonical = canonical_expression(req.expression)

    inputs_record = [
        {
            "name": s.name,
            "value": s.value,
            "unit": f"{declared_units[s.name]:~}",
            "standard_uncertainty": s.standard_uncertainty,
            "distribution": s.distribution,
        }
        for s in req.inputs
    ]
    correlations_record = _pairs(req.correlations)

    input_dimensions = {
        s.name: format_dimensionality(declared_units[s.name].dimensionality)
        for s in req.inputs
    }
    # 用输出单位推回输出维度（确保与表达式一致）
    output_dimensionality = format_dimensionality(parse_unit(req.output_unit).dimensionality)

    summary = {
        "model_version": req.version,
        "expression": expr_canonical,
        "expression_sha256": sha256_hex(expr_canonical),
        "variables": list(compiled.variables),
        "inputs": sorted(
            [
                {
                    "name": s.name,
                    "unit": f"{declared_units[s.name]:~}",
                    "dimensionality": input_dimensions[s.name],
                    "distribution": s.distribution,
                }
                for s in req.inputs
            ],
            key=lambda d: d["name"],
        ),
        "output_unit": req.output_unit,
        "output_dimensionality": output_dimensionality,
        "correlations": correlations_record,
        "whitelist": {
            "operators": ["+", "-", "*", "/", "**", "unary+", "unary-"],
            "functions": [
                "sqrt", "abs", "exp", "log", "log10",
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                "sinh", "cosh", "tanh",
            ],
            "constants": ["pi", "e"],
        },
    }

    row = {
        "version": req.version,
        "name": req.name,
        "description": req.description,
        "expression": req.expression,
        "expression_canonical": expr_canonical,
        "inputs": inputs_record,
        "output_unit": req.output_unit,
        "correlations": correlations_record,
        "summary": summary,
    }
    storage.insert_model(row)
    _CONTEXT_CACHE[req.version] = (compiled, ctx)
    return storage.get_model(req.version)


def _get_context(version: str) -> tuple[Any, Any]:
    cached = _CONTEXT_CACHE.get(version)
    if cached is not None:
        return cached
    record = storage.get_model(version)
    names = tuple(s["name"] for s in record["inputs"])
    compiled = compile_expression(record["expression"], names)
    declared_units = {s["name"]: parse_unit(s["unit"]) for s in record["inputs"]}
    ctx = build_unit_context(compiled, declared_units, record["output_unit"])
    _CONTEXT_CACHE[version] = (compiled, ctx)
    return compiled, ctx


# ---------------------------------------------------------------------------
# 计算请求
# ---------------------------------------------------------------------------

def _match_value(a: float, b: float) -> bool:
    return abs(a - b) <= max(_VALUE_MATCH_ABS, _VALUE_MATCH_REL * max(abs(a), abs(b)))


def _reconcile_snapshot(req: EvaluationCreate, record: dict[str, Any]) -> dict[str, Any]:
    """把请求输入快照换算到模型声明单位，并与登记的不可变模型逐项核对。

    返回规范化快照（声明单位下的数值）。单位允许等价写法（如 ``V`` 与 ``mV``），
    其余字段（值、标准不确定度、分布、相关系数）必须与登记一致。
    """
    req_by_name = {s.name: s for s in req.inputs}
    reg_names = [s["name"] for s in record["inputs"]]
    if set(req_by_name) != set(reg_names):
        missing = sorted(set(reg_names) - set(req_by_name))
        extra = sorted(set(req_by_name) - set(reg_names))
        raise RequestError(f"输入快照与模型不一致（缺失 {missing}，多余 {extra}）")

    declared_units = {s["name"]: parse_unit(s["unit"]) for s in record["inputs"]}
    normalized: list[dict[str, Any]] = []
    for reg in record["inputs"]:
        snap = req_by_name[reg["name"]]
        unit = declared_units[reg["name"]]
        try:
            value_decl = to_declared_magnitude(snap.value, snap.unit, unit)
            std_decl = to_declared_magnitude(snap.standard_uncertainty, snap.unit, unit)
        except RequestError:
            raise

        if not _match_value(value_decl, reg["value"]):
            raise RequestError(
                f"输入 {reg['name']!r} 的名义值与登记模型不符: "
                f"快照换算后 {value_decl!r}，登记 {reg['value']!r}"
            )
        if not _match_value(std_decl, reg["standard_uncertainty"]):
            raise RequestError(
                f"输入 {reg['name']!r} 的标准不确定度与登记模型不符: "
                f"快照换算后 {std_decl!r}，登记 {reg['standard_uncertainty']!r}"
            )
        if snap.distribution != reg["distribution"]:
            raise RequestError(
                f"输入 {reg['name']!r} 的分布与登记模型不符: "
                f"{snap.distribution} != {reg['distribution']}"
            )
        normalized.append({
            "name": reg["name"],
            "value": reg["value"],
            "unit": f"{unit:~}",
            "standard_uncertainty": reg["standard_uncertainty"],
            "distribution": reg["distribution"],
        })

    normalized.sort(key=lambda d: d["name"])

    # 相关系数核对（规范化为有序对集合）
    req_pairs = {frozenset((p.i, p.j)): round(float(p.r), 12) for p in req.correlations}
    reg_pairs = {frozenset((t[0], t[1])): float(t[2]) for t in record["correlations"]}
    if set(req_pairs) != set(reg_pairs):
        raise RequestError("相关系数对的集合与登记模型不一致")
    for key, r in req_pairs.items():
        if not _match_value(r, reg_pairs[key]):
            raise RequestError(f"相关系数 {sorted(key)} 与登记模型不符: {r} != {reg_pairs[key]}")

    return {
        "inputs": normalized,
        "correlations": copy.deepcopy(record["correlations"]),
    }


def submit_evaluation(req: EvaluationCreate) -> dict[str, Any]:
    record = storage.get_model(req.model_version)  # 不存在 -> 404
    compiled, ctx = _get_context(req.model_version)

    # 1. 快照与不可变模型核对（含单位换算、维度兼容）
    snapshot = _reconcile_snapshot(req, record)

    # 2. 请求指纹：模型版本 + 输入快照 + N + 种子 + 容差 + 幂等键
    fingerprint = {
        "model_version": req.model_version,
        "snapshot": snapshot,
        "n": req.n,
        "seed": req.seed,
        "convergence_tolerance": req.convergence_tolerance,
        "idempotency_key": req.idempotency_key,
    }
    request_hash = sha256_hex(canonical_json(fingerprint))

    # 3. 幂等快路径
    prior = storage.find_evaluation_by_idempotency_key(req.idempotency_key)
    if prior is not None:
        if prior["request_hash"] != request_hash:
            raise ConflictError(
                f"幂等键 {req.idempotency_key!r} 已绑定到不同的计算请求（evaluation {prior['id']}）"
            )
        prior["cached"] = True
        return prior

    names = tuple(s["name"] for s in record["inputs"])
    distributions = tuple(
        next(s["distribution"] for s in record["inputs"] if s["name"] == name)
        for name in names
    )

    # 4. 基本单位空间
    x_decl = {s["name"]: s["value"] for s in record["inputs"]}
    std_decl = {s["name"]: s["standard_uncertainty"] for s in record["inputs"]}
    x_base = ctx.nominal_to_base(x_decl)
    std_base_arr = np.array([ctx.std_to_base(std_decl)[name] for name in names], dtype=float)
    zero_std = {name for name in names if std_decl[name] == 0.0}

    corr = build_correlation_matrix(
        names, [(t[0], t[1], float(t[2])) for t in record["correlations"]], zero_std
    )

    # 5. GUM 一阶
    gum = run_gum(compiled, names, x_base, std_base_arr, corr, ctx)

    # 6. 蒙特卡洛（固定种子）；不收敛 -> 抛错，不保存任何结果
    mc = run_monte_carlo(
        compiled=compiled,
        names=names,
        distributions=distributions,
        x_base=x_base,
        std_base=std_base_arr,
        corr=corr,
        n_draws=req.n,
        seed=req.seed,
        ctx=ctx,
        convergence_tol=req.convergence_tolerance,
    )
    if not mc.converged:
        from .errors import ConvergenceError

        raise ConvergenceError(
            f"蒙特卡洛 N={req.n} 未通过收敛判据（可增大 N 后以新幂等键重试）: "
            + "; ".join(mc.nonconvergence_reasons),
        )
    mc.convergence["seed"] = float(req.seed)

    # 7. 组装结果（仅在计算全部成功后写库，且写库是唯一的“保存”动作）
    result = {
        "model_version": req.model_version,
        "model_summary": record["summary"],
        "method": {
            "gum": "GUM 一阶线性化（Guide to the Expression of Uncertainty in Measurement）",
            "monte_carlo": "GUM Supplement 1 自适应风格蒙特卡洛（Gaussian copula + 边际分布）",
            "coverage": 0.95,
        },
        "snapshot": snapshot,
        "parameters": {
            "n": req.n,
            "seed": req.seed,
            "convergence_tolerance": req.convergence_tolerance,
        },
        "combined": {
            "gum": {
                "value": _f(gum.y),
                "standard_uncertainty": _f(gum.uc),
                "coverage_interval": {"low": _f(gum.coverage_low), "high": _f(gum.coverage_high)},
                "coverage_factor": _f(gum.k),
                "unit": req_output_unit(record),
            },
            "monte_carlo": {
                "mean": _f(mc.mean),
                "standard_uncertainty": _f(mc.std),
                "median": _f(mc.median),
                "coverage_interval": {"low": _f(mc.coverage_low), "high": _f(mc.coverage_high)},
                "samples_range": {"min": _f(mc.samples_min), "max": _f(mc.samples_max)},
                "unit": req_output_unit(record),
            },
        },
        "sensitivities": {
            name: {"coefficient": _f(c), "unit": f"{record['output_unit']}/{s_unit(name, record)}"}
            for name, c in gum.sensitivities.items()
        },
        "contributions": {
            "gum": [
                {
                    "source": c["input"],
                    "kind": c["kind"],
                    "variance_contribution": _f(float(c["variance_contribution"])),
                    "fraction": _f(float(c["fraction"])),
                }
                for c in gum.contributions
            ],
            "monte_carlo": [
                {
                    "source": d["input"],
                    "pearson_r": _f(d["pearson_r"]),
                    "r_squared": _f(d["r_squared"]),
                }
                for d in mc.input_correlations
            ],
        },
        "convergence": {**{k: _f(v) for k, v in mc.convergence.items()}, "converged": True},
    }

    eval_id = uuid.uuid4().hex
    stored = storage.insert_evaluation(
        eval_id=eval_id,
        request_hash=request_hash,
        idempotency_key=req.idempotency_key,
        model_version=req.model_version,
        inputs_json=canonical_json(snapshot),
        n=req.n,
        seed=req.seed,
        tol=req.convergence_tolerance,
        result_json=canonical_json(result),
    )
    out = stored["evaluation"]
    out["cached"] = stored["cached"]
    return out


def req_output_unit(record: dict[str, Any]) -> str:
    return record["output_unit"]


def s_unit(name: str, record: dict[str, Any]) -> str:
    return next(s["unit"] for s in record["inputs"] if s["name"] == name)


def _f(x: Any) -> float:
    """统一转 float（剔除 numpy 标量，保证 JSON 可序列化）。

    常规尺度沿用历史实现的 **15 位小数** 舍入（固定种子流的既有响应
    逐位不变）；但固定小数位会把小量级模型的真实结果（如方差贡献
    1e-18）直接舍成 0.0，因此对 |x| < 1e-15 的小值改用 **15 位有效
    数字** 舍入——与数值尺度无关，保留真实小正方差的响应精度。
    """
    v = float(x)
    if v == 0.0 or not math.isfinite(v) or abs(v) >= 1e-15:
        return round(v, 15)
    with localcontext() as ctx:
        ctx.prec = 15
        return float(+Decimal(v))


# ---------------------------------------------------------------------------
# 测量精度反求（precision budget）
# ---------------------------------------------------------------------------

#: 名义点灵敏度是否判零直接采用引擎逐分量的可辨结论
#: （numerical_gradient_with_noise 返回的 resolved 标记）：中心差分的差分值
#: |f(x+h)-f(x-h)| == |g_k|·2h 不超过输出舍入尺度 eps·max|f±| 的
#: GRADIENT_RESOLUTION_FACTOR 倍时，梯度被偏置舍入淹没，判为零
#: （如 2*(a-a) 恰为 0）。自适应步长先放大到中心差分最优尺度
#: h_cap = eps^(1/3)·max(|x0|,1) 再裁决：输出带大偏置的真实非零梯度
#: （如 y = 1 + 1e-4·b, b=0）在放大后的步长上差分值可辨，不再被误判为零；
#: 扩步被定义域边界截断时（如 1.0 + 1e-10*sqrt(x) 在 x0=1e-8 处 x-h<0，
#: 步长到不了 h_cap），改用域内单侧探针放大到 h_cap 确认输出确实依赖该
#: 输入，真实斜率（~5e-7）不再被边界处不可辨的中心差分误判为零
#: （真零方向如 0.0*sqrt(x) 单侧探针到 h_cap 仍不可辨，维持零结论）。
#: 逐输入按自身尺度裁决，绝不与其他输入的灵敏度量级比较（那会把 1e-13
#: 级的真实微小灵敏度误判为零）。


def _unit_label(spec: str) -> str:
    """pint 紧凑格式把无量纲单位渲染成空串；响应文本统一显示为 ``dimensionless``。"""
    return "dimensionless" if spec == "" else spec


def solve_precision_budget(req: PrecisionBudgetCreate) -> dict[str, Any]:
    """反求单个输入标准不确定度的可行区间，使 GUM 合成标准不确定度不超过目标。

    只读不可变模型，不改模型、不写任何计算记录：

    * 沿用登记时的单位换算与量纲校验（目标单位须与输出量纲一致）；
    * 沿用登记模型的相关矩阵（重新走同一套半正定校验）；
    * 沿用 GUM 名义点灵敏度（同一套中心差分）；
    * 仅令被求输入 k 的标准差 u 自由变化（u ≥ 0），其余输入保持登记值。

    合成方差关于 u 是二次函数（协方差全部纳入，含负相关项）：

        v(u) = C + B·u + A·u²

        A = g_k²
        B = 2·g_k·Σ_{j≠k} g_j·ρ_kj·s_j
        C = Σ_{i≠k}Σ_{j≠k} g_i g_j Cov(i,j)

    （g、s 均在基本单位空间；目标平方先换算到输出方差空间再除以
    output_scale²。）求解 v(u) ≤ t²、u ≥ 0 的区间，并把端点换算回
    该输入的**声明单位**。
    """
    record = storage.get_model(req.model_version)  # 未知版本 -> 404
    compiled, ctx = _get_context(req.model_version)

    names = tuple(s["name"] for s in record["inputs"])
    if req.input_name not in names:
        raise RequestError(
            f"输入 {req.input_name!r} 未在模型版本 {req.model_version!r} 中登记，可选: {list(names)}"
        )
    k = names.index(req.input_name)

    # 1. 目标单位换算与量纲校验（与输出声明单位量纲一致）
    out_unit = parse_unit(record["output_unit"])
    target_out = _target_to_output(
        req.target.unit, req.target.standard_uncertainty, out_unit, record["output_unit"]
    )
    t_base_sq = (target_out / ctx.output_scale) ** 2
    # 2. 名义点（声明单位 -> 基本单位）与 GUM 名义点灵敏度
    x_decl = {s["name"]: s["value"] for s in record["inputs"]}
    std_decl = {s["name"]: s["standard_uncertainty"] for s in record["inputs"]}
    x_base = ctx.nominal_to_base(x_decl)
    std_base = np.array([ctx.std_to_base(std_decl)[name] for name in names], dtype=float)

    zero_std = {name for name in names if std_decl[name] == 0.0}
    corr = build_correlation_matrix(
        names, [(t[0], t[1], float(t[2])) for t in record["correlations"]], zero_std
    )

    grad, _grad_noise, grad_resolved = numerical_gradient_with_noise(compiled, x_base, names)
    if not np.all(np.isfinite(grad)):
        raise PropagationError("名义点邻域产生了非有限结果（NaN/Inf），无法进行精度反求")

    gk = float(grad[k])

    # 当前登记预算下的合成方差（与 GUM 传播同一二次型）与当前标准不确定度
    cov_cur = std_base[:, None] * corr * std_base[None, :]
    cur_var_base = float(grad @ cov_cur @ grad)
    cur_uc_out = ctx.output_scale * math.sqrt(max(cur_var_base, 0.0))

    # 3. 二次型系数（u 为输入 k 的标准差，基本单位空间）
    A = gk * gk
    other = [j for j in range(len(names)) if j != k]
    B = 2.0 * gk * float(sum(grad[j] * corr[k, j] * std_base[j] for j in other))
    C = float(sum(
        grad[i] * grad[j] * std_base[i] * std_base[j] * corr[i, j]
        for i in other for j in other
    ))
    # 浮点噪声钳位（与 run_gum 同尺度思想：C 理论上是 PSD 二次型，≥0）
    cov0_scale = float(sum(
        abs(grad[i] * grad[j] * std_base[i] * std_base[j] * corr[i, j])
        for i in other for j in other
    ))
    c_floor = _EPS * max(1.0, len(names)) * cov0_scale
    if -c_floor <= C <= 0.0:
        C = 0.0

    # 4. 端点（先在基本单位空间求解）
    # 零灵敏度裁决：直接采用引擎逐分量的可辨结论（见模块顶部注释）——
    # 与该输入梯度自身的舍入误差尺度比较（随问题尺度收缩，小量级模型的
    # 真实微小灵敏度如 1e-13 必须保留），定义域边界截断时经域内单侧探针
    # 确认依赖（真实斜率如 5e-7 不得误判为零），绝不与其他输入的灵敏度
    # 量级比较。
    zero_g = gk == 0.0 or not bool(grad_resolved[k])
    # 与 k 相关（登记矩阵中存在边）时，u=0 端点无效：相关系数在零方差处无定义
    correlated = any(names[k] in (t[0], t[1]) for t in record["correlations"])

    empty = False
    unbounded = False
    lo_base = hi_base = 0.0
    lo_open = hi_open = False

    if zero_g:
        # 输入 k 对输出零灵敏度：方差与 u 无关 => 可行时无上限，否则空区间
        if C > t_base_sq + c_floor:
            empty = True
        else:
            unbounded = True
            lo_open = correlated  # 相关输入的零端点无效 => 开
    else:
        disc = B * B - 4.0 * A * (C - t_base_sq)
        # 舍入误差级负判别式（理论相切）钳为 0
        disc_scale = max(abs(B * B), abs(4.0 * A * (C - t_base_sq)), 1e-300)
        if disc < 0.0 and disc >= -_EPS * max(1.0, len(names)) * disc_scale:
            disc = 0.0
        if disc < 0.0:
            # v(u) 恒大于目标平方（开口向上且无实根）=> 无论 u 取何值都无法达标
            empty = True
        else:
            # 实根存在：可行域为 [u_-, u_+] 与 u ≥ 0 的交。
            # 根的舍入容差与根自身量级同阶（无任何绝对下限）：小量级模型的
            # 真实小正根（如 1e-12 m）必须保留，不能被绝对阈值吸到 0。
            root_eps = _EPS13 * max(math.sqrt(max(disc, 0.0)) / (2.0 * A),
                                    abs(B) / (2.0 * A))
            u_minus = (-B - math.sqrt(max(disc, 0.0))) / (2.0 * A)
            u_plus = (-B + math.sqrt(max(disc, 0.0))) / (2.0 * A)
            if u_plus < -root_eps:
                empty = True  # 整个可行区间落在 u < 0（理论上 B≤0 时不应发生，防御式）
            else:
                # 与 u ≥ 0 取交
                if abs(u_minus) <= root_eps:
                    u_minus = 0.0
                if abs(u_plus) <= root_eps:
                    u_plus = 0.0
                lo_base = max(u_minus, 0.0)
                hi_base = max(u_plus, 0.0)
                lo_open = correlated and lo_base == 0.0  # 相关输入的零端点无效 => 开
                # 二次方程根端点恒可取（闭）
                # 相切（u_- == u_+）时浮点可能给出极窄区间，塌缩为单点
                if 0.0 < lo_base <= hi_base and hi_base - lo_base <= 2.0 * root_eps:
                    mid = 0.5 * (lo_base + hi_base)
                    lo_base = hi_base = mid
                # 唯一候选 u=0 又因相关而无效 => 可行集为空（如 r=+1 相切于原点）
                if hi_base == 0.0 and lo_open:
                    empty = True
                    lo_open = False

    # 5. 端点换算回输入声明单位
    input_record = next(s for s in record["inputs"] if s["name"] == req.input_name)
    decl_unit_spec = input_record["unit"]
    factor = ctx.base_factors[req.input_name]  # 1 声明单位 = factor 个基本单位

    interval: dict[str, Any]
    if empty:
        interval = {
            "feasible": False,
            "lower": None,
            "lower_inclusive": False,
            "upper": None,
            "upper_inclusive": False,
            "unbounded_upper": False,
        }
    elif unbounded:
        interval = {
            "feasible": True,
            "lower": 0.0,
            "lower_inclusive": not lo_open,
            "upper": None,
            "upper_inclusive": False,
            "unbounded_upper": True,
        }
    else:
        lo_decl = _tidy(lo_base / factor)
        hi_decl = _tidy(hi_base / factor)
        if lo_decl > hi_decl:  # 舍入倒置保护
            hi_decl = lo_decl
        interval = {
            "feasible": True,
            "lower": _f(lo_decl),
            "lower_inclusive": not lo_open,
            "upper": _f(hi_decl),
            "upper_inclusive": not hi_open,
            "unbounded_upper": False,
        }

    # 6. 当前登记值是否达标（尺度相关容差，与 GUM 钳零思想一致）
    meet_tol = _EPS ** 0.5 * max(cur_uc_out, target_out, 0.0)
    satisfied = cur_var_base >= 0.0 and cur_uc_out <= target_out + meet_tol

    sensitivity_out = ctx.sensitivity(req.input_name, gk)
    out_label = _unit_label(record["output_unit"])
    in_label = _unit_label(decl_unit_spec)
    return {
        "model_version": req.model_version,
        "input_name": req.input_name,
        "input_unit": in_label,
        "target": {
            "standard_uncertainty": _f(req.target.standard_uncertainty),
            "unit": _unit_label(req.target.unit),
        },
        "target_in_output_unit": {
            "standard_uncertainty": _f(target_out),
            "unit": out_label,
        },
        "sensitivity": {
            "coefficient": _f(sensitivity_out),
            "unit": f"{out_label}/{in_label}",
        },
        "current_standard_uncertainty": {
            "value": _f(input_record["standard_uncertainty"]),
            "unit": in_label,
        },
        "current_combined_standard_uncertainty": {
            "value": _f(cur_uc_out),
            "unit": out_label,
        },
        "satisfied": bool(satisfied),
        "interval": interval,
    }


def _target_to_output(unit_spec: str, value: float, out_unit: pint.Unit, out_spec: str) -> float:
    """把目标值换算到模型输出声明单位；量纲不符 -> 422。"""
    reg = get_registry()
    src = parse_unit(unit_spec)
    try:
        q = reg.Quantity(float(value), src).to(out_unit)
    except pint.errors.DimensionalityError as exc:
        raise RequestError(
            f"目标单位 {unit_spec!r} 与模型输出单位 {out_spec!r} 量纲不兼容"
        ) from exc
    return float(q.magnitude)


def _tidy(v: float) -> float:
    """端点理论上非负：仅把 -0.0 与（防御式的）负值归一到 0。

    真实的小正端点（如 1e-9 mm）必须原样保留——零值与舍入误差的裁决
    已在基本单位空间按根的量级完成，这里绝不能再套用与单位尺度无关的
    绝对阈值（旧实现会把任何小于 ~6e-6 的正端点错误清零）。
    """
    if v <= 0.0:
        return 0.0
    return v
