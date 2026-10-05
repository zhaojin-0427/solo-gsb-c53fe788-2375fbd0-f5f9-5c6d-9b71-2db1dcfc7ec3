"""不确定度传播计算引擎：GUM 一阶线性化 + 蒙特卡洛。

所有数值计算在调用方给定的 *基本单位量纲空间* 中完成（见 :mod:`app.units`），
引擎本身不感知单位，只感知数值、标准差与相关系数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .errors import CorrelationError, PropagationError, ConvergenceError
from .expressions import CompiledExpression, numpy_evaluation

# ---------------------------------------------------------------------------
# 分布
# ---------------------------------------------------------------------------

#: 支持的输入量分布
DISTRIBUTIONS = ("normal", "uniform", "triangular")

_EPS = np.finfo(float).eps
_EPS13 = _EPS ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 相关矩阵
# ---------------------------------------------------------------------------

#: 半正定判据：最小特征值低于该阈值视为明显非 PSD，必须拒绝
PSD_EIG_TOL = 1e-8


def build_correlation_matrix(
    names: tuple[str, ...],
    pairs: list[tuple[str, str, float]],
    zero_std: set[str],
) -> np.ndarray:
    """由 (i, j, r) 对构造并校验相关系数矩阵。

    * 对称、对角为 1、|r| <= 1；
    * 不允许引用标准不确定度为 0 的输入量（相关系数在那里无定义）；
    * 必须半正定（允许奇异：完全相关 r = ±1 时零特征值是合法的）；
      最小特征值低于 :data:`PSD_EIG_TOL`（-1e-8）明显非半正定则拒绝；
      处于 [-1e-8, 0) 的微小负特征值视为舍入误差，投影到最近相关矩阵
      （半正定 + 保持单位对角）后放行。
    """
    n = len(names)
    idx = {name: k for k, name in enumerate(names)}
    rmat = np.eye(n)

    for a, b, r in pairs:
        if a == b:
            raise CorrelationError(f"输入量 {a!r} 不能与自身指定相关系数")
        if a in zero_std or b in zero_std:
            raise CorrelationError(
                f"相关系数 ({a!r}, {b!r}) 引用了标准不确定度为 0 的输入量"
            )
        if not -1.0 <= r <= 1.0:
            # 允许浮点序列化造成的 1±1e-12 级别越界，明显越界仍拒绝
            if abs(r) <= 1.0 + 1e-12:
                r = max(-1.0, min(1.0, r))
            else:
                raise CorrelationError(f"相关系数 ({a!r}, {b!r}) 越界: r={r}")
        ia, ib = idx[a], idx[b]
        if rmat[ia, ib] != 0.0:
            raise CorrelationError(f"相关系数 ({a!r}, {b!r}) 重复指定")
        rmat[ia, ib] = r
        rmat[ib, ia] = r

    if n == 0:
        return rmat

    eig = np.linalg.eigvalsh(rmat)
    # 奇异半正定矩阵（如 r 恰为 ±1 时存在零特征值）是合法的，不要求正定；
    # 仅把最小特征值低于 -1e-8 的明显非 PSD 矩阵判为非法。
    if eig[0] < -PSD_EIG_TOL:
        worst = eig[0]
        raise CorrelationError(
            f"相关矩阵不是半正定矩阵：最小特征值 {worst:.3e} < -{PSD_EIG_TOL:.0e}"
        )

    # 浮点钳位：仅 [-1e-8, 0) 的舍入误差级负特征值需要修复。
    # 用 Higham 交替投影把矩阵投影到相关矩阵锥（保对角 1 + 半正定）。
    # 注意：最小特征值恰为 0 的奇异矩阵（完全相关）保持原样，不做修复。
    if eig[0] < 0.0:
        rmat = _nearest_correlation_matrix(rmat)

    # 终检：对称、单位对角、半正定（允许零特征值），不再要求 Cholesky 可分解
    rmat = 0.5 * (rmat + rmat.T)
    np.fill_diagonal(rmat, 1.0)
    final_min = float(np.min(np.linalg.eigvalsh(rmat)))
    if final_min < -1e-12:
        raise CorrelationError(
            f"相关矩阵修复后仍不是半正定矩阵：最小特征值 {final_min:.3e}"
        )
    return rmat


def correlation_factor(corr: np.ndarray) -> np.ndarray:
    """返回因子 F，使相关标准正态 Z = F W（W 为独立标准正态）且 F F^T = corr。

    正定矩阵直接走 Cholesky（与历史实现逐位一致，固定种子流不变）；
    奇异半正定矩阵（含合法零特征值，如 r = ±1 的完全相关）Cholesky 会失败，
    此时改用特征分解的对称平方根 F = Q sqrt(Λ) Q^T：

    * r = +1 时两列完全相同，两个标准化输入同向变化；
    * r = -1 时两列正好反号，两个标准化输入反向变化。
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(corr)
        if float(np.min(evals)) < -1e-10:
            raise CorrelationError("相关矩阵不是半正定矩阵，无法构造抽样因子")
        evals = np.clip(evals, 0.0, None)
        factor = (evecs * np.sqrt(evals)) @ evecs.T
        return 0.5 * (factor + factor.T)


def _align_perfectly_correlated(z: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """对相关系数恰为 ±1 的输入对，强制标准化正态行严格同向/反向。

    奇异因子在高阶完全相关块上可能留下 ~1e-15 的特征向量正交化残差；
    用带权并查集沿 ``r = ±1`` 的边传播符号后，r = +1 的两行逐位相同、
    r = -1 的两行逐位相反。正态边际直接使用 z（不经 CDF/分位往返近似），
    故 y = a + b 的两个标准化输入严格同向或反向变化，r = -1 且 ua = ub 时
    输出严格为常数。

    仅作用于矩阵中**精确存储**为 ±1.0 的边：非奇异矩阵与经舍入修复的
    近奇异矩阵（相关系数为 1±δ）逐位不变。
    """
    n = corr.shape[0]
    perfect = [
        (i, j, float(corr[i, j]))
        for i in range(n) for j in range(i + 1, n)
        if corr[i, j] == 1.0 or corr[i, j] == -1.0
    ]
    if not perfect:
        return z

    parent = list(range(n))
    weight = [1.0] * n  # z_i = weight[i] * z_root（分量根为 parent 链顶）

    def root_sign(i: int) -> tuple[int, float]:
        s = 1.0
        while parent[i] != i:
            s *= weight[i]
            i = parent[i]
        return i, s

    for i, j, rij in perfect:
        ri, si = root_sign(i)
        rj, sj = root_sign(j)
        if ri == rj:
            continue  # 符号矛盾的块最小特征值为 -2 量级，已在 PSD 校验拒绝
        # z_i = si·z_ri = rij·z_j = rij·sj·z_rj  =>  z_rj = rij·si·sj·z_ri
        parent[rj] = ri
        weight[rj] = rij * si * sj

    out = z
    for i in range(n):
        ri, si = root_sign(i)
        if ri != i:
            if out is z:
                out = z.copy()
            out[i] = si * z[ri]  # 根行从不被改写，引用的始终是原始行
    return out


def _nearest_correlation_matrix(a: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Dykstra 交替投影求最近相关矩阵（Higham 2002 简化版）。"""
    n = a.shape[0]
    y = a.copy()
    delta = np.zeros_like(a)
    for _ in range(max_iter):
        # 投影 1：谱半正定投影
        evals, evecs = np.linalg.eigh(y - delta)
        x = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
        # Dykstra 修正量
        delta = x - (y - delta)
        # 投影 2：对角归一化（单位对角）
        d = np.sqrt(np.diag(x))
        d[d == 0] = 1.0
        y = x / np.outer(d, d)
        np.fill_diagonal(y, 1.0)
        if np.min(np.linalg.eigvalsh(y)) >= -1e-12:
            try:
                np.linalg.cholesky(y)
                return y
            except np.linalg.LinAlgError:
                pass
    return y


# ---------------------------------------------------------------------------
# 边际分布抽样（Gaussian copula，相关结构正确且边际分布精确）
# ---------------------------------------------------------------------------

def _phi(z: np.ndarray) -> np.ndarray:
    """标准正态 CDF（Abramowitz & Stegun 26.2.17，绝对误差 < 7.5e-8）。"""
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * az)
    poly = (((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t)
    tail = poly * np.exp(-0.5 * az * az) / math.sqrt(2.0 * math.pi)
    return np.where(z >= 0.0, 1.0 - tail, tail)


def _marginal_unit_samples(distribution: str, u: np.ndarray) -> np.ndarray:
    """把均匀概率 u 映射为 *标准差为 1* 的目标边际分布样本。"""
    if distribution == "normal":
        # 标准正态分位数：Acklam 近似（绝对误差 ~1.15e-9）
        return _norm_ppf(u)
    if distribution == "uniform":
        # U(-sqrt(3), sqrt(3))：标准差为 1 的均匀分布
        return math.sqrt(3.0) * (2.0 * u - 1.0)
    if distribution == "triangular":
        # 对称三角(-1,0,1)，标准差 1/sqrt(6)，缩放后标准差为 1
        out = np.empty_like(u)
        lo = u <= 0.5
        out[lo] = -1.0 + np.sqrt(2.0 * u[lo])
        out[~lo] = 1.0 - np.sqrt(2.0 * (1.0 - u[~lo]))
        return math.sqrt(6.0) * out
    raise PropagationError(f"不支持的分布: {distribution}")  # pragma: no cover


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """标准正态逆 CDF（Peter Acklam 近似）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)

    m1 = p < plow
    q = np.sqrt(-2.0 * np.log(p[m1]))
    out[m1] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    m3 = p > phigh
    q = np.sqrt(-2.0 * np.log(1.0 - p[m3]))
    out[m3] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    m2 = ~(m1 | m3)
    q = p[m2] - 0.5
    r = q * q
    out[m2] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
              (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


# ---------------------------------------------------------------------------
# 数值微分
# ---------------------------------------------------------------------------

def _evaluate_scalar(compiled: CompiledExpression, env_base: Mapping[str, float]) -> float:
    y = numpy_evaluation(compiled, env_base)
    yv = float(np.asarray(y, dtype=float))
    if not math.isfinite(yv):
        raise PropagationError("模型在名义点（或其邻域）产生了非有限结果（NaN/Inf），无法传播不确定度")
    return yv


def numerical_gradient(compiled: CompiledExpression, x_base: Mapping[str, float], names: tuple[str, ...]) -> np.ndarray:
    """中心差分灵敏度 dy/dx（基本单位空间）。"""
    grad, _, _ = numerical_gradient_with_noise(compiled, x_base, names)
    return grad


#: 名义点灵敏度判定为“零”的噪声倍数：中心差分差分值 |f(x+h)-f(x-h)|
#: （等价地 |g_k|·2h）不超过输出舍入尺度的该倍数时，视为被输出偏置的
#: 舍入噪声淹没（如 2*(a-a) 恰为 0）。倍数 >1 只为覆盖多步运算累积的
#: 舍入。services 的零灵敏度裁决必须复用同一倍数，保证“差分不清”与
#: “梯度判零”两条路径口径一致。
GRADIENT_RESOLUTION_FACTOR = 100.0

#: 零/近零名义点处差分步长向上放大的几何因子；上限 h_cap = eps^(1/3) 是
#: 中心差分平衡截断误差 O(h²) 与舍入误差 O(eps/h) 的经典最优步长尺度。
_GRADIENT_STEP_GROWTH = 8.0

#: 可辨后继续放大步长时，相邻梯度允许的单步相对漂移。线性模型的舍入
#: 噪声随 h 增大而减小（漂移恒小）；而截断误差 O(h²) 每放大 8 倍步长
#: 增长 64 倍（漂移巨大）——该因子介于两者之间，用于在高曲率零名义点
#: 回退到漂移前的小步长。
_GRADIENT_STABILITY = 0.5

#: 对称探针在初始 h0 就越过定义域时，按该倍数收缩步长直到两侧都落在
#: 有效域内（如 asin(1-1e-8) 的边缘距离只有 1e-8，远小于 h0≈6e-6）。
_GRADIENT_STEP_SHRINK = 8.0
_GRADIENT_STEP_SHRINK_MAX = 16

#: 定义域截断对称扩步时，域内单侧探针网格的参数：
#: 以域内最大有限对称步长 h_edge 为锚点、按 ×2 几何级数向开阔侧布点；
#: 对单侧差商 D_m = (f(x0+s·2^m·h_edge)-f(x0))/(s·2^m·h_edge) 做
#: Richardson 外推（消去 O(h)、O(h²) 截断项）。sqrt 这类 x^(1/2) 在
#: 域边缘的 Taylor 级数全阶非零，外推阶数不宜高：取到二阶且当对角
#: 相邻级相对漂移超过 _ONESIDED_STABILITY 时停在上一级（噪声接管）。
_ONESIDED_LEVELS = 10
_ONESIDED_MAX_ORDER = 2
_ONESIDED_STABILITY = 0.25


def _try_scalar(compiled: CompiledExpression, env_base: Mapping[str, float]) -> float | None:
    """求值并返回 float；结果非有限（NaN/Inf）或求值失败时返回 ``None``。

    用于定义域敏感的探针：探针越过模型定义域（如 sqrt/log 的负数自变量、
    asin 的 |x|>1）时返回 None，由调用方向内收缩/换侧，而不是直接抛错。
    """
    try:
        with np.errstate(all="ignore"):
            yv = float(np.asarray(numpy_evaluation(compiled, env_base), dtype=float))
    except (PropagationError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(yv):
        return None
    return yv


def _probe_pair(compiled, x_base: Mapping[str, float], name: str, x0: float, step: float):
    """对称探针 (f(x0+h), f(x0-h))；任一侧越界/非有限则返回 None。"""
    env_plus = dict(x_base)
    env_minus = dict(x_base)
    env_plus[name] = x0 + step
    env_minus[name] = x0 - step
    yp = _try_scalar(compiled, env_plus)
    ym = _try_scalar(compiled, env_minus)
    if yp is None or ym is None:
        return None
    return yp, ym


def _richardson_onesided(Ds: list[float]) -> tuple[float, int]:
    """单侧等比（因子 2）差商的 Richardson 外推，沿外推表第 0 列定阶。

    D_m 对单侧差分含全阶截断误差 D(h) = g + c1·h + c2·h² + …；第 r 级
    递推 T_r[i] = (2^r·T_{r-1}[i] - T_{r-1}[i+1])/(2^r - 1) 依次消去
    O(h^r) 项。sqrt 型函数在边缘处级数收敛慢、高阶外推还会放大舍入，
    故相对相邻级的改善一旦低于 :data:`_ONESIDED_STABILITY` 即停止。
    """
    g = Ds[0]
    order = 0
    prev = Ds
    for r in range(1, _ONESIDED_MAX_ORDER + 1):
        if len(prev) < 2:
            break
        qr = 2.0 ** r
        nxt = [(qr * prev[i] - prev[i + 1]) / (qr - 1) for i in range(len(prev) - 1)]
        cur = nxt[0]
        if abs(cur - g) > _ONESIDED_STABILITY * max(abs(cur), abs(g)):
            break
        g, order, prev = cur, r, nxt
    return g, order


def _richardson_l1(order: int) -> float:
    """第 ``order`` 级第 0 列外推相对原始差商的系数 L1 和（舍入放大倍数）。

    递推 T_r[i] = (q·T_{r-1}[i] - T_{r-1}[i+1])/(q-1)（q = 2^r）在第 0
    列对原始 D_0…D_r 的系数满足 c_0←q/(q-1)·c_0，c_j←q/(q-1)·c_j
    −1/(q-1)·c_{j-1}（j=1..r-1），c_r←−1/(q-1)·c_{r-1}。
    """
    coeffs = [1.0]
    for r in range(1, order + 1):
        qr = 2.0 ** r
        a = qr / (qr - 1.0)
        b = -1.0 / (qr - 1.0)
        nxt = [0.0] * (len(coeffs) + 1)
        for j, c in enumerate(coeffs):
            nxt[j] += a * c
            nxt[j + 1] += b * c
        coeffs = nxt
    return sum(abs(c) for c in coeffs)


def _onesided_domain_fallback(
    compiled: CompiledExpression,
    x_base: Mapping[str, float],
    name: str,
    x0: float,
    h_last: float,
    h_cap: float,
    f0: float,
) -> tuple[float, float, bool]:
    """对称扩步被模型定义域截断时的域内单侧探针回退。

    背景：近定义域边界的名义点（如 y = 1 + 1e-10·sqrt(x)，x0 = 1e-8），
    对称探针一旦 h > x0 就因 x0-h < 0 越界，步长被压在远小于
    ``h_cap = eps^(1/3)`` 的尺度上；偏置 1.0 的舍入（1 ulp ≈ 2.2e-16）
    把真实斜率 5e-7 的差分值完全淹没，百倍门槛判零。但向**开阔侧**
    （sqrt 即 +h 方向）探针不受边界限制，可以走到差分值远超舍入尺度
    的步长上确认斜率真实非零，再以 Richardson 外推消去单侧差分的
    O(h)、O(h²) 曲率截断，回收可信斜率。

    返回 ``(gradient, noise, resolved)``：

    * 先用 ×2 步行把“域内最大有限对称步长”收紧到真实边缘的 2 倍以内；
    * 以该步长为锚点分别向两侧布单侧网格，取可延伸更远（更开阔）的一侧；
    * ``resolved`` 只要任一原始探针的差分超过百倍输出舍入尺度即为真——
      真零斜率（如 1 + 0·sqrt(x)）所有探针恒等，仍判零；
    * 不可辨时梯度严格归 0，``noise`` 为外推系数 L1 放大后的舍入尺度。
    """
    # 1. 定位域边缘：从最后一个有限对称步长 ×2 收紧（两次相邻尝试间 ≤2×）
    h_edge = h_last
    while h_edge < h_cap:
        hn = h_edge * 2.0
        if _probe_pair(compiled, x_base, name, x0, hn) is None:
            break
        h_edge = hn

    # 2. 分别向两侧布单侧网格，选开阔侧（可延伸的有限探针层数更多）
    def grid(side: int) -> tuple[list[float], list[float], int, float]:
        hs: list[float] = []
        Ds: list[float] = []
        n_resolved = 0
        fscale = abs(f0)
        h = h_edge
        for _ in range(_ONESIDED_LEVELS):
            env = dict(x_base)
            env[name] = x0 + side * h
            fp = _try_scalar(compiled, env)
            if fp is None:
                break
            fscale = max(fscale, abs(fp))
            hs.append(h)
            Ds.append((fp - f0) / (side * h))
            if abs(fp - f0) > GRADIENT_RESOLUTION_FACTOR * _EPS * max(abs(fp), abs(f0)):
                n_resolved += 1
            h *= 2.0
        return hs, Ds, n_resolved, fscale

    grids = {side: grid(side) for side in (+1, -1)}
    # 开阔侧：有限层数多者优先；层数相同（如两侧都开阔）时可辨探针多者优先
    side = max((+1, -1), key=lambda s: (len(grids[s][0]), grids[s][2]))
    hs, Ds, n_resolved, fscale = grids[side]

    if not Ds:
        return 0.0, _EPS * abs(f0) / max(h_edge, _EPS), False

    # 3. Richardson 外推定阶（至少 2 层；仅 1 层时只能用原始差商，
    # 外推系数 L1 退化为 1，即普通单侧差分的舍入尺度）
    if len(Ds) >= 2:
        g, order = _richardson_onesided(Ds)
    else:
        g, order = Ds[0], 0
    noise = _EPS * fscale * _richardson_l1(order) / hs[0]

    # 4. 零/非零裁决只看原始探针是否可辨（外推值在零函数上也可能偶然非零）
    if n_resolved == 0:
        return 0.0, noise, False
    return g, noise, True


def numerical_gradient_with_noise(
    compiled: CompiledExpression, x_base: Mapping[str, float], names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """中心差分灵敏度、逐分量舍入误差尺度与逐分量“可辨”标志（基本单位空间）。

    返回 ``(grad, noise, resolved)``。``noise[k]`` 估计函数值舍入（~eps·|f|）
    经 ``(f(x+h)-f(x-h))/(2h)`` 传播到第 k 个梯度分量的量级：

        noise[k] = eps · max(|f(x+h)|, |f(x-h)|) / (2h)

    它随问题自身尺度收缩（小量级模型的 noise 同样小）。``resolved[k]``
    表示第 k 个方向上存在超过输出舍入尺度百倍门槛的真实差分信号——
    “零灵敏度”裁决应直接用它（逐输入、按自身尺度），绝不能与其他输入
    的灵敏度量级比较。

    **输出偏置自适应步长。** 名义点为 0（或极小）时初始步长
    ``h0 = eps^(2/3)`` 也极小；若输出在名义点带大偏置（如 y = 1 + 1e-4·b，
    b = 0），差分真值 g·2h 可能只有偏置 ulp 的几分之一，被舍入噪声淹没，
    使真实非零梯度在百倍噪声门槛下误判为零。因此逐分量从 h0 起按 8 倍
    几何放大步长，直到差分值在输出舍入尺度上**可辨**
    （``|f+−f-| > F·eps·max|f±|``）或到达步长上限
    ``h_cap = eps^(1/3)·max(|x0|, 1)``（中心差分最优步长尺度）。
    初始 h0 即可辨时直接采用（与历史实现逐位一致，不额外求值）。

    可辨后仍继续放大并比较相邻步长的梯度：线性模型的舍入噪声随步长
    缩小（漂移恒小），故一路放大到 h_cap 获得高精度梯度；一旦单步相对
    漂移超过 :data:`_GRADIENT_STABILITY`（截断误差 O(h²) 随 8 倍步长
    增长 64 倍的特征），回退采用漂移前的小步长，避免在高曲率零名义点
    把截断伪影当成灵敏度。

    **初始探针即越界（h0 越过定义域）。** 如 asin(1-1e-8) 的边缘距离
    只有 1e-8 ≪ h0，log(x0) 在 x0-h≤0 处无定义：按 8 倍向内收缩步长
    直到两侧探针都落在有效域内（至多 :data:`_GRADIENT_STEP_SHRINK_MAX`
    次），再走正常的放大/稳定流程；收缩后仍无可用探针（如名义点本身
    就在 sqrt/log 的开域边界 x0=0）才抛错——与历史行为一致。

    **定义域截断扩步（近边界误报的修复路径）。** 放大后的步长越过模型
    定义域（如 sqrt 在极小正名义点处 x-h < 0 产生 NaN）时，对称扩步
    被迫停在远小于 h_cap 的步长上；若此时差分仍被输出偏置的舍入淹没，
    转入 :func:`_onesided_domain_fallback`：定位域边缘，以域内最大对称
    步长为锚点向开阔侧按 ×2 布单侧探针，用原始探针的可辨性区分真实
    斜率与真零/偏置舍入，并以 Richardson 外推消除单侧差分的曲率截断
    误差，回收可信梯度（如 y = 1 + 1e-10·sqrt(x)，x0 = 1e-8 的真实
    斜率 5e-7）。真正零/无扰动方向（f+ == f- 恒成立，如 ``2*(a-a)``、
    ``1 + 0*sqrt(x)``）所有探针差分为零，resolved=False、梯度保持 0.0。
    """
    grad = np.zeros(len(names))
    noise = np.zeros(len(names))
    resolved_flag = np.zeros(len(names), dtype=bool)
    for k, name in enumerate(names):
        x0 = x_base[name]
        h = _EPS13 * max(abs(x0), _EPS13)
        h_cap = _EPS13 * max(abs(x0), 1.0)

        # 初始探针：对称步长（可能极小或越过定义域）
        pair = _probe_pair(compiled, x_base, name, x0, h)
        initial_blocked = pair is None
        if initial_blocked:
            pair = None
            for _ in range(_GRADIENT_STEP_SHRINK_MAX):
                h /= _GRADIENT_STEP_SHRINK
                pair = _probe_pair(compiled, x_base, name, x0, h)
                if pair is not None:
                    break
            if pair is None:
                # 名义点本身就在开域边界（如 sqrt(0)/log(0)）：历史行为抛错
                raise PropagationError(
                    "模型在名义点（或其邻域）产生了非有限结果（NaN/Inf），无法传播不确定度"
                )
        yp, ym = pair
        g = (yp - ym) / (2.0 * h)
        resolved = abs(yp - ym) > GRADIENT_RESOLUTION_FACTOR * _EPS * max(abs(yp), abs(ym))

        # 阶段 1：差分值被输出舍入淹没 => 放大步长直到可辨或到达 h_cap。
        # 初始 h0 即越界后又被收缩的情形（initial_blocked）不进入扩步
        # 阶段 2 的高精度放大——它的“名义尺度”是边缘距离而非 eps^(1/3)，
        # 继续放大立刻撞墙（log 在 x0-h≤0）或被曲率接管（asin 在 x=1）。
        grew = False
        domain_blocked = False
        if not resolved:
            while not resolved and h < h_cap:
                hn = min(h * _GRADIENT_STEP_GROWTH, h_cap)
                nxt = _probe_pair(compiled, x_base, name, x0, hn)
                if nxt is None:
                    domain_blocked = True
                    break  # 放大越过定义域：保留当前有限步长
                h, (yp, ym) = hn, nxt
                g = (yp - ym) / (2.0 * h)
                resolved = abs(yp - ym) > GRADIENT_RESOLUTION_FACTOR * _EPS * max(abs(yp), abs(ym))
                grew = True

        # 阶段 2（仅当偏置迫使步长放大过、且不是被定义域逼停）：继续放大
        # 到 h_cap 进一步压低偏置导致的舍入误差；一旦梯度出现截断性漂移
        # （或越过定义域）即回退/停止。h0 已可辨时不进入本阶段——初始
        # 极小步长上恰被浮点精确表示的梯度（如 sin(0)）与历史实现逐位一致。
        if resolved and grew and not domain_blocked and not initial_blocked:
            while h < h_cap:
                hn = min(h * _GRADIENT_STEP_GROWTH, h_cap)
                nxt = _probe_pair(compiled, x_base, name, x0, hn)
                if nxt is None:
                    break  # 越过定义域：保留当前有限步长
                ypn, ymn = nxt
                gn = (ypn - ymn) / (2.0 * hn)
                rn = abs(ypn - ymn) > GRADIENT_RESOLUTION_FACTOR * _EPS * max(abs(ypn), abs(ymn))
                if not rn or abs(gn - g) > _GRADIENT_STABILITY * max(abs(gn), abs(g)):
                    break  # 截断接管（或可辨性丧失）：保留当前 (h, g, yp, ym)
                h, yp, ym, g = hn, ypn, ymn, gn

        if not resolved and domain_blocked:
            # 对称扩步被定义域截断且差分仍不可辨：域内单侧探针回退。
            # f0 此刻必然有限——否则初始探针阶段就已抛错。
            env0 = dict(x_base)
            env0[name] = x0
            f0 = _try_scalar(compiled, env0)
            if f0 is None:
                raise PropagationError(
                    "模型在名义点（或其邻域）产生了非有限结果（NaN/Inf），无法传播不确定度"
                )
            g, g_noise, g_resolved = _onesided_domain_fallback(
                compiled, x_base, name, x0, h, h_cap, f0
            )
            grad[k] = g
            noise[k] = g_noise
            resolved_flag[k] = g_resolved
            continue

        grad[k] = g
        noise[k] = _EPS * max(abs(yp), abs(ym)) / (2.0 * h)
        resolved_flag[k] = bool(resolved)
    return grad, noise, resolved_flag


# ---------------------------------------------------------------------------
# GUM 一阶
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GumResult:
    y: float
    uc: float
    coverage_low: float
    coverage_high: float
    k: float
    sensitivities: dict[str, float]
    contributions: list[dict[str, float | str]]


def run_gum(
    compiled: CompiledExpression,
    names: tuple[str, ...],
    x_base: dict[str, float],
    std_base: np.ndarray,
    corr: np.ndarray,
    ctx,
) -> GumResult:
    """GUM 一阶线性化（输出空间）。"""
    y_base = _evaluate_scalar(compiled, x_base)
    grad_base = numerical_gradient(compiled, x_base, names)
    cov = (std_base[:, None] * corr * std_base[None, :])
    var_y = float(grad_base @ cov @ grad_base)
    # 舍入误差尺度必须与实际参与求和的协方差项同阶：
    #   各项 g_i g_j Cov(i,j) 的绝对值之和（对角项 + 两倍成对项），
    #   再按项数放宽，以同时覆盖矩阵乘法舍入与中心差分梯度（相对误差
    #   ~eps^(1/3)）在“理论完全抵消”时留下的残差。
    # 绝不能使用与输入尺度无关的固定绝对阈值（旧实现含 max(1.0, ...)），
    # 否则小量级模型（如 u = 1e-9，真实方差 4e-18）会被误判为舍入噪声
    # 而钳为零，导致标准不确定度、覆盖区间与逐项贡献全部错误归零。
    term_scale = float(
        np.abs(grad_base[:, None] * grad_base[None, :] * cov).sum()
    )
    var_floor = _EPS * max(1.0, float(cov.shape[0])) * term_scale
    # 完全相关抵消（r = -1、梯度与标准差逐项匹配）时，理论方差恰为 0；
    # 浮点运算会残留正/负噪声，统一钳为 0。阈值随协方差项尺度收缩，
    # 真实小正方差（uc 远大于 0）不受影响。
    if -var_floor <= var_y <= var_floor:
        var_y = 0.0
    if var_y < 0:
        raise PropagationError(f"GUM 合成方差为负 ({var_y:.3e})")
    uc_base = math.sqrt(var_y)

    y = float(ctx.from_base(y_base))
    uc = ctx.output_scale * uc_base
    coverage_factor = 1.959963984540054  # 95% 正态覆盖因子
    sensitivities = {name: ctx.sensitivity(name, float(grad_base[ki])) for ki, name in enumerate(names)}

    # 方差贡献分解：对角项 + 协方差对
    contributions: list[dict[str, float | str]] = []
    total_var = var_y if var_y > 0 else 1.0
    for ki, name in enumerate(names):
        v = grad_base[ki] ** 2 * cov[ki, ki]
        contributions.append({
            "input": name,
            "variance_contribution": (ctx.output_scale ** 2) * v,
            "fraction": v / total_var if var_y > 0 else 0.0,
            "kind": "individual",
        })
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = 2.0 * grad_base[i] * grad_base[j] * cov[i, j]
            if v != 0.0:
                contributions.append({
                    "input": f"{names[i]}~{names[j]}",
                    "variance_contribution": (ctx.output_scale ** 2) * v,
                    "fraction": v / total_var if var_y > 0 else 0.0,
                    "kind": "covariance",
                })

    return GumResult(
        y=y,
        uc=uc,
        coverage_low=y - coverage_factor * uc,
        coverage_high=y + coverage_factor * uc,
        k=coverage_factor,
        sensitivities=sensitivities,
        contributions=contributions,
    )


# ---------------------------------------------------------------------------
# 蒙特卡洛
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class McResult:
    mean: float
    std: float
    median: float
    coverage_low: float
    coverage_high: float
    samples_min: float
    samples_max: float
    input_correlations: list[dict[str, float]]
    convergence: dict[str, float]
    converged: bool
    nonconvergence_reasons: list[str]


def _gaussian_density(x: float, samples: np.ndarray, sd: float) -> float:
    """Silverman 高斯核密度估计（用于分位标准误）。"""
    n = samples.size
    h = 1.06 * sd * n ** (-0.2) if sd > 0 else 1.0
    if h == 0:
        return 0.0
    z = (x - samples) / h
    return float(np.exp(-0.5 * z * z).sum() / (n * h * math.sqrt(2.0 * math.pi)))


def run_monte_carlo(
    compiled: CompiledExpression,
    names: tuple[str, ...],
    distributions: tuple[str, ...],
    x_base: dict[str, float],
    std_base: np.ndarray,
    corr: np.ndarray,
    n_draws: int,
    seed: int,
    ctx,
    convergence_tol: float,
) -> McResult:
    """N 次蒙特卡洛传播 + 收敛统计。"""
    rng = np.random.default_rng(seed)
    n = len(names)
    means = np.array([x_base[name] for name in names], dtype=float)

    # 相关标准正态：Z = F W。奇异半正定相关矩阵（r = ±1 完全相关）同样可抽样：
    # r = +1 时对应行逐位相同（同向变化），r = -1 时恰好反号（反向变化）。
    lower = correlation_factor(corr)
    w = rng.standard_normal(size=(n, n_draws))
    z = lower @ w
    # r 恰为 ±1 的完全相关边：强制标准化正态严格同向/反向（消除分解残差）
    has_perfect = bool(np.any(np.abs(corr - np.eye(n)) >= 1.0))
    if has_perfect:
        z = _align_perfectly_correlated(z, corr)

    # Gaussian copula：标准正态 -> 均匀概率 -> 目标边际分位数。
    # 正态边际在历史实现中也走 phi/ppf 往返；为保持非奇异请求的固定种子流
    # 逐位不变，这里默认沿用该路径。仅当相关结构含 r = ±1 完全相关边时，
    # 正态边际直接使用 z（z 本身就是精确标准正态样本）——phi/ppf 往返近似
    # 会在完全反相（r = -1）时留下残差噪声，使本该严格抵消的 y = a + b
    # 不抵消，并可能把退化（零方差）输出误判为不收敛。
    # 非正态边际（均匀/三角）始终经 CDF 变换以保持精确边际分布。
    u = _phi(z)
    u = np.clip(u, 1e-12, 1 - 1e-12)  # 数值保护

    draws = np.empty((n, n_draws))
    for k, dist in enumerate(distributions):
        if dist == "normal" and has_perfect:
            draws[k] = means[k] + std_base[k] * z[k]
        else:
            draws[k] = means[k] + std_base[k] * _marginal_unit_samples(dist, u[k])

    env = {name: draws[k] for k, name in enumerate(names)}
    # 抽样可能越过模型定义域（如 sqrt/log），numpy 会给出 NaN + RuntimeWarning；
    # 下面立即统计非有限值并明确拒绝，告警本身无意义，压掉避免污染日志。
    with np.errstate(all="ignore"):
        y_base = numpy_evaluation(compiled, env)
    y_base = np.asarray(y_base, dtype=float)

    if np.any(~np.isfinite(y_base)):
        bad = int(np.sum(~np.isfinite(y_base)))
        raise PropagationError(
            f"蒙特卡洛抽样中模型产生了 {bad}/{n_draws} 个非有限结果（NaN/Inf）；"
            "通常意味着输入分布越过了模型定义域（如对负数开方/取对数）"
        )

    y = ctx.from_base(y_base)
    mean = float(np.mean(y))
    sd = float(np.std(y, ddof=1))
    median = float(np.median(y))
    q_lo, q_hi = (float(v) for v in np.quantile(y, [0.025, 0.975]))

    # ---- 逐输入相关性诊断（输出方差的相关份额 r^2） ----
    input_correlations: list[dict[str, float]] = []
    if sd > 0:
        yc = y - mean
        y_var = float(yc @ yc)
        for k, name in enumerate(names):
            xk = draws[k]
            if np.std(xk) > 0:
                xc = xk - xk.mean()
                r = float(xc @ yc / math.sqrt(float(xc @ xc) * y_var))
            else:
                r = 0.0
            input_correlations.append({"input": name, "pearson_r": r, "r_squared": r * r})
    else:
        input_correlations = [{"input": name, "pearson_r": 0.0, "r_squared": 0.0} for name in names]

    # ---- 收敛统计 ----
    kurt = float(((y - mean) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else 0.0
    se_mean = sd / math.sqrt(n_draws)
    se_sd = sd * math.sqrt(max(kurt + 2.0, 0.0) / (4.0 * n_draws)) if sd > 0 else 0.0
    half_width = max(q_hi - q_lo, _EPS)
    if sd > 0:
        f_lo = _gaussian_density(q_lo, y, sd)
        f_hi = _gaussian_density(q_hi, y, sd)
        f_lo = max(f_lo, 1e-12)
        f_hi = max(f_hi, 1e-12)
        se_q_lo = math.sqrt(0.025 * 0.975 / n_draws) / f_lo
        se_q_hi = math.sqrt(0.025 * 0.975 / n_draws) / f_hi
        se_q = max(se_q_lo, se_q_hi)
    else:
        se_q_lo = se_q_hi = se_q = 0.0

    denom_mean = max(abs(mean), sd)  # 均值接近 0 时以标准差为尺度，避免相对误差发散
    rel_mean = se_mean / denom_mean
    rel_sd = se_sd / sd if sd > 0 else 0.0
    rel_q = se_q / half_width

    reasons: list[str] = []
    if sd > 0:
        if rel_mean > convergence_tol:
            reasons.append(f"均值标准误相对值 {rel_mean:.3e} 超过容差 {convergence_tol:g}")
        if rel_sd > convergence_tol:
            reasons.append(f"标准差标准误相对值 {rel_sd:.3e} 超过容差 {convergence_tol:g}")
        if rel_q > convergence_tol:
            reasons.append(f"95% 分位标准误相对区间半宽 {rel_q:.3e} 超过容差 {convergence_tol:g}")
    converged = not reasons

    convergence = {
        "n": float(n_draws),
        "seed": float(0),  # 占位，由调用方覆盖（保持字段集中）
        "kurtosis_excess": kurt,
        "se_mean": se_mean,
        "se_std": se_sd,
        "se_quantile_low": se_q_lo,
        "se_quantile_high": se_q_hi,
        "rel_se_mean": rel_mean,
        "rel_se_std": rel_sd,
        "rel_se_quantile": rel_q,
        "tolerance": convergence_tol,
    }

    return McResult(
        mean=mean,
        std=sd,
        median=median,
        coverage_low=q_lo,
        coverage_high=q_hi,
        samples_min=float(np.min(y)),
        samples_max=float(np.max(y)),
        input_correlations=input_correlations,
        convergence=convergence,
        converged=converged,
        nonconvergence_reasons=reasons,
    )
