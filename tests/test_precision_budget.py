"""测量精度反求 POST /api/v1/precision-budgets 的端到端测试。"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import storage


@pytest.fixture(scope="module")
def client():
    storage.init_db()
    with TestClient(app) as c:
        yield c


def _sum_model(version: str, ua: float, ub: float, r):
    return {
        "version": version,
        "expression": "a + b",
        "inputs": [
            {"name": "a", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": ua, "distribution": "normal"},
            {"name": "b", "value": 5.0, "unit": "dimensionless",
             "standard_uncertainty": ub, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [] if r is None else [{"i": "a", "j": "b", "r": r}],
    }


def _register(client, model):
    r = client.post("/api/v1/models", json=model)
    assert r.status_code == 201, r.text


def _budget(client, version, name, target, unit="dimensionless"):
    return client.post("/api/v1/precision-budgets", json={
        "model_version": version,
        "input_name": name,
        "target": {"standard_uncertainty": target, "unit": unit},
    })


# ---------------------------------------------------------------------------
# 核心二次不等式（含负相关）
# ---------------------------------------------------------------------------

def test_negative_correlation_interval_is_bounded_both_sides(client):
    # y = a + b, r = -1, ub = 2, 目标 1 => (ua-2)^2 <= 1 => ua ∈ [1, 3]
    # 绝不能把负相关误算成单一上限（那会错误得到 [0, 3]）。
    _register(client, _sum_model("pb-rm1", 0.1, 2.0, -1.0))
    r = _budget(client, "pb-rm1", "a", 1.0)
    assert r.status_code == 200, r.text
    body = r.json()
    iv = body["interval"]
    assert iv["feasible"] is True
    assert iv["lower_inclusive"] is True and iv["upper_inclusive"] is True
    assert abs(iv["lower"] - 1.0) < 1e-9
    assert abs(iv["upper"] - 3.0) < 1e-9
    assert iv["unbounded_upper"] is False
    # 当前 ua=0.1 => uc = 1.9 > 1，不达标
    assert body["satisfied"] is False
    assert abs(body["current_combined_standard_uncertainty"]["value"] - 1.9) < 1e-8
    assert body["input_unit"] == "dimensionless"
    assert body["sensitivity"]["unit"] == "dimensionless/dimensionless"
    assert abs(body["sensitivity"]["coefficient"] - 1.0) < 1e-9


def test_endpoints_actually_satisfy_and_violate(client):
    # 端点处恰达标，区间外必超标（针对 [1,3] 用传播管线交叉验证）
    def uc_at(ua):
        return abs(ua - 2.0)

    for ua in (1.0, 2.0, 3.0):
        assert uc_at(ua) <= 1.0 + 1e-12
    for ua in (0.5, 0.999, 3.001, 5.0):
        assert uc_at(ua) > 1.0


def test_positive_correlation_unreachable_returns_empty(client):
    # r = +1, ub = 2：v(u) = (u+2)^2 的最小值 4 > 目标平方 1 => 空区间
    _register(client, _sum_model("pb-rp1-empty", 0.1, 2.0, 1.0))
    r = _budget(client, "pb-rp1-empty", "a", 1.0)
    assert r.status_code == 200
    iv = r.json()["interval"]
    assert iv == {
        "feasible": False,
        "lower": None,
        "lower_inclusive": False,
        "upper": None,
        "upper_inclusive": False,
        "unbounded_upper": False,
    }
    assert r.json()["satisfied"] is False


def test_independent_input_zero_endpoint_closed_and_satisfied(client):
    # y = a + b, r = 0, ub = 1, 目标 sqrt(2) => ua^2 <= 1 => [0, 1]，零端点闭
    _register(client, _sum_model("pb-indep", 0.5, 1.0, None))
    r = _budget(client, "pb-indep", "a", math.sqrt(2.0))
    assert r.status_code == 200
    iv = r.json()["interval"]
    assert iv["feasible"] is True
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert abs(iv["upper"] - 1.0) < 1e-9 and iv["upper_inclusive"] is True
    # 当前 ua=0.5 => uc = sqrt(1.25) <= sqrt(2)，达标
    assert r.json()["satisfied"] is True


def test_correlated_input_zero_endpoint_open(client):
    # y = a + b, r = -1, ub = 0.5, 目标 1 => |ua - 0.5| <= 1 => ua ∈ (0, 1.5]
    # u=0 与 r=-1 的登记相关结构冲突（不能引用零标准差输入），故零端点开。
    _register(client, _sum_model("pb-open-zero", 0.5, 0.5, -1.0))
    r = _budget(client, "pb-open-zero", "a", 1.0)
    iv = r.json()["interval"]
    assert iv["feasible"] is True
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is False
    assert abs(iv["upper"] - 1.5) < 1e-9 and iv["upper_inclusive"] is True


def test_tangency_at_zero_with_correlation_is_empty(client):
    # r = +1, ub = 1, 目标 1：唯一可行 u=0，但相关输入的零端点无效 => 空区间
    _register(client, _sum_model("pb-tangent-zero", 0.1, 1.0, 1.0))
    iv = _budget(client, "pb-tangent-zero", "a", 1.0).json()["interval"]
    assert iv["feasible"] is False and iv["lower"] is None and iv["upper"] is None


def test_b_other_input_with_zero_std_correlated(client):
    # 调节 b（与 a 相关），a 自身 ua=0.3 非零：区间 (0, 0.7]（零端点开）
    _register(client, _sum_model("pb-rm1-b", 0.3, 0.1, -1.0))
    r = _budget(client, "pb-rm1-b", "b", 0.4)
    iv = r.json()["interval"]
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is False
    assert abs(iv["upper"] - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# 零灵敏度：无上限，上界 null
# ---------------------------------------------------------------------------

def test_zero_sensitivity_independent_unbounded_closed_zero(client):
    # y = 2*(a-a) + b：a 的灵敏度恰为 0；b 独立 ub=0.2 => 任意 ua 可行，[0, null]
    model = {
        "version": "pb-flat-indep",
        "expression": "2*(a - a) + b",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 2.0, "unit": "dimensionless",
             "standard_uncertainty": 0.2, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-flat-indep", "a", 0.3).json()
    assert body["sensitivity"]["coefficient"] == 0.0
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is True
    assert iv["upper"] is None and iv["upper_inclusive"] is False
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert body["satisfied"] is True  # 当前 uc = 0.2 ≤ 0.3


def test_zero_sensitivity_correlated_open_zero(client):
    # 同上但 a、b 相关 r=0.5：零端点无效 => (0, null)
    model = {
        "version": "pb-flat-corr",
        "expression": "2*(a - a) + b",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 2.0, "unit": "dimensionless",
             "standard_uncertainty": 0.2, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [{"i": "a", "j": "b", "r": 0.5}],
    }
    _register(client, model)
    iv = _budget(client, "pb-flat-corr", "a", 0.3).json()["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is True
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is False


def test_zero_sensitivity_target_unreachable_empty(client):
    # a 零灵敏度但 b 的固定贡献 0.2 已超过目标 0.1 => 空区间
    body = _budget(client, "pb-flat-indep", "a", 0.1).json()
    assert body["interval"]["feasible"] is False
    assert body["satisfied"] is False


# ---------------------------------------------------------------------------
# 目标 0、单输入、单位换算
# ---------------------------------------------------------------------------

def test_zero_target_single_input(client):
    model = {
        "version": "pb-zero-target",
        "expression": "2*a",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-zero-target", "a", 0.0).json()
    assert body["interval"] == {
        "feasible": True, "lower": 0.0, "lower_inclusive": True,
        "upper": 0.0, "upper_inclusive": True, "unbounded_upper": False,
    }
    assert body["satisfied"] is False  # 当前 ua=0.1 => uc=0.2


def test_target_zero_allows_tightening_current_model(client):
    # 当前值正好达标（ua=0, 单输入）
    model = {
        "version": "pb-zero-ok",
        "expression": "a",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.0, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-zero-ok", "a", 0.0).json()
    assert body["satisfied"] is True


def test_unit_conversion_endpoints_in_declared_unit(client):
    # 单输入 y = L，模型声明单位 millimeter；目标 0.5 centimeter = 5 mm => u ∈ [0, 5] mm
    model = {
        "version": "pb-length",
        "expression": "L",
        "inputs": [
            {"name": "L", "value": 1000.0, "unit": "millimeter",
             "standard_uncertainty": 1.0, "distribution": "normal"},
        ],
        "output_unit": "millimeter",
        "correlations": [],
    }
    _register(client, model)
    r = _budget(client, "pb-length", "L", 0.5, "centimeter")
    assert r.status_code == 200, r.text
    body = r.json()
    # 输入单位按登记存储的紧凑规格回显；输出换算保留输出单位的规格字符串
    assert body["input_unit"] == "mm"
    assert body["target_in_output_unit"] == {"standard_uncertainty": 5.0, "unit": "millimeter"}
    iv = body["interval"]
    assert iv["lower"] == 0.0 and abs(iv["upper"] - 5.0) < 1e-9
    assert body["sensitivity"]["unit"] == "millimeter/mm"


def test_power_model_with_correlation_unit_target(client):
    # P = V*I, r(V,I)=0.5：vary V。v = 4uV² + 2*2*10*0.5*0.02*uV + 100*0.0004
    #                              = 4u² + 0.4u + 0.04
    model = {
        "version": "pb-power",
        "expression": "V * I",
        "inputs": [
            {"name": "V", "value": 10.0, "unit": "V",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "I", "value": 2.0, "unit": "A",
             "standard_uncertainty": 0.02, "distribution": "normal"},
        ],
        "output_unit": "W",
        "correlations": [{"i": "V", "j": "I", "r": 0.5}],
    }
    _register(client, model)
    r = _budget(client, "pb-power", "V", 0.5, "W")
    assert r.status_code == 200
    iv = r.json()["interval"]
    # 4u² + 0.4u + 0.04 = 0.25  =>  u_+ = (-0.4 + sqrt(0.16+3.36))/8
    expected = (-0.4 + math.sqrt(3.52)) / 8
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is False  # 相关 => 零端点开
    assert abs(iv["upper"] - expected) < 1e-9
    assert r.json()["satisfied"] is True


# ---------------------------------------------------------------------------
# 拒绝场景
# ---------------------------------------------------------------------------

def test_unknown_version_404(client):
    r = _budget(client, "pb-does-not-exist", "a", 1.0)
    assert r.status_code == 404
    assert r.json()["error"] == "NotFoundError"


def test_unknown_input_422(client):
    r = _budget(client, "pb-rm1", "ghost", 1.0)
    assert r.status_code == 422
    assert "ghost" in r.json()["message"]


def test_target_dimension_mismatch_422(client):
    r = _budget(client, "pb-rm1", "a", 1.0, "meter")
    assert r.status_code == 422
    assert "量纲不兼容" in r.json()["message"]


def test_target_bad_unit_422(client):
    r = _budget(client, "pb-rm1", "a", 1.0, "frobnicate")
    assert r.status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_target_422_valid_json(client, literal):
    r = client.post(
        "/api/v1/precision-budgets",
        content=('{"model_version":"pb-rm1","input_name":"a",'
                 f'"target":{{"standard_uncertainty":{literal},"unit":"dimensionless"}}}}'),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert isinstance(r.json(), list)  # 错误响应本身必须是合法 JSON


def test_negative_target_422(client):
    r = _budget(client, "pb-rm1", "a", -0.5)
    assert r.status_code == 422


def test_missing_target_field_422(client):
    r = client.post("/api/v1/precision-budgets",
                    json={"model_version": "pb-rm1", "input_name": "a"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 小量级反求：真实微小灵敏度/小正端点不得被绝对阈值误判为零
# ---------------------------------------------------------------------------

def test_tiny_scale_length_closed_interval(client):
    # y = L（毫米），L = 0、uL = 2e-9 mm，目标 1e-9 mm => u ∈ [0, 1e-9] mm 闭区间。
    # 旧实现把 1e-12 m 的根按绝对阈值（~3.7e-11）吸到 0、_tidy 再把 1e-9 mm 清零。
    model = {
        "version": "pb-tiny-length",
        "expression": "L",
        "inputs": [
            {"name": "L", "value": 0.0, "unit": "millimeter",
             "standard_uncertainty": 2e-9, "distribution": "normal"},
        ],
        "output_unit": "millimeter",
        "correlations": [],
    }
    _register(client, model)
    r = _budget(client, "pb-tiny-length", "L", 1e-9, "millimeter")
    assert r.status_code == 200, r.text
    body = r.json()
    iv = body["interval"]
    assert iv["feasible"] is True
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True   # 独立输入零端点闭
    assert iv["upper"] == 1e-9 and iv["upper_inclusive"] is True  # 真实小正上界保留
    assert iv["unbounded_upper"] is False
    # 当前 uL = 2e-9 > 1e-9，不达标
    assert body["satisfied"] is False
    assert abs(body["current_combined_standard_uncertainty"]["value"] - 2e-9) < 1e-20


def test_tiny_sensitivity_not_misjudged_as_zero(client):
    # y = a + 1e-13*b（无量纲），a = b = 0，ua = 0、ub = 2e9，目标 1e-4，反求 b。
    # 真实灵敏度 1e-13（远小于 a 的 1.0）不得被相对阈值判为零：
    # v(u) = (1e-13*u)^2 <= 1e-8 => u ∈ [0, 1e9]。
    model = {
        "version": "pb-tiny-sens",
        "expression": "a + 1e-13*b",
        "inputs": [
            {"name": "a", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 0.0, "distribution": "normal"},
            {"name": "b", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 2e9, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    r = _budget(client, "pb-tiny-sens", "b", 1e-4)
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(body["sensitivity"]["coefficient"] - 1e-13) < 1e-20
    iv = body["interval"]
    assert iv["feasible"] is True
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert abs(iv["upper"] - 1e9) / 1e9 < 1e-9 and iv["upper_inclusive"] is True
    assert iv["unbounded_upper"] is False
    # 当前 ub = 2e9 => uc = 2e-4 > 1e-4，不达标
    assert body["satisfied"] is False
    assert abs(body["current_combined_standard_uncertainty"]["value"] - 2e-4) < 1e-15


def test_tiny_scale_negative_correlation_two_sided(client):
    # 小量级 + 负相关：y = a + b，r = -1，ub = 2e-9，目标 1e-9
    # => (ua - 2e-9)^2 <= 1e-18 => ua ∈ [1e-9, 3e-9]，双端有界且端点不丢。
    _register(client, _sum_model("pb-tiny-rm1", 1e-10, 2e-9, -1.0))
    r = _budget(client, "pb-tiny-rm1", "a", 1e-9)
    assert r.status_code == 200, r.text
    iv = r.json()["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is False
    assert abs(iv["lower"] - 1e-9) < 1e-20 and iv["lower_inclusive"] is True
    assert abs(iv["upper"] - 3e-9) < 1e-20 and iv["upper_inclusive"] is True


def test_tiny_scale_zero_sensitivity_still_unbounded(client):
    # 小量级模型中的真零灵敏度（2*(a-a)）仍判为零：可行时无上限。
    model = {
        "version": "pb-tiny-flat",
        "expression": "2*(a - a) + b",
        "inputs": [
            {"name": "a", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 1e-9, "distribution": "normal"},
            {"name": "b", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 1e-9, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-tiny-flat", "a", 1e-8).json()
    assert body["sensitivity"]["coefficient"] == 0.0
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is True
    assert iv["upper"] is None
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert body["satisfied"] is True


# ---------------------------------------------------------------------------
# 输出偏置：小真实灵敏度被偏置舍入淹没时，自适应差分步长必须恢复梯度
# ---------------------------------------------------------------------------

def test_biased_output_small_sensitivity_is_bounded(client):
    # y = a + 1e-4*b（无量纲），a=1（带输出偏置）、b=0，ua=0、ub=10000，
    # 目标 0.5，反求 b。初始极小步长 eps^(2/3) 上差分真值只有偏置 1.0 的
    # ~33 ulp，被百倍噪声门槛误判为零 -> 旧实现错误返回无上界。
    # 结合步长、偏置与舍入自适应放大步长后梯度可辨（≈1e-4）：
    # v(u) = (1e-4*u)^2 <= 0.25 => u ∈ [0, 5000]，闭区间，当前值不达标。
    model = {
        "version": "pb-bias-small-sens",
        "expression": "a + 1e-4*b",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.0, "distribution": "normal"},
            {"name": "b", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 10000.0, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    r = _budget(client, "pb-bias-small-sens", "b", 0.5)
    assert r.status_code == 200, r.text
    body = r.json()
    # 灵敏度恢复到 1e-4（相对误差远小于旧实现被淹没时的 ~1.4%）
    assert abs(body["sensitivity"]["coefficient"] - 1e-4) < 1e-9
    assert body["sensitivity"]["unit"] == "dimensionless/dimensionless"
    iv = body["interval"]
    assert iv["feasible"] is True
    assert iv["unbounded_upper"] is False
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert abs(iv["upper"] - 5000.0) < 1e-3 and iv["upper_inclusive"] is True
    # 当前 ub = 10000 => uc ≈ 1.0 > 0.5，不达标
    assert body["satisfied"] is False
    assert abs(body["current_combined_standard_uncertainty"]["value"] - 1.0) < 1e-6


def test_biased_output_nonlinear_slope_recovered(client):
    # 偏置 + 非线性：y = 1 + sin(x)，x=0 处真实斜率 1；小步长上偏置 1 的
    # 舍入主导。斜率在初始步长上即可辨（约 2 万 ulp），区间仍须有界正确。
    model = {
        "version": "pb-bias-sin",
        "expression": "1.0 + sin(x)",
        "inputs": [
            {"name": "x", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-bias-sin", "x", 0.05).json()
    assert abs(body["sensitivity"]["coefficient"] - 1.0) < 1e-5
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is False
    assert abs(iv["upper"] - 0.05) < 1e-5


def test_biased_output_true_zero_slope_still_unbounded(client):
    # 偏置下真零斜率（1 + x*x 在 x=0）仍判零：任意 ux 可行 => 无上界。
    model = {
        "version": "pb-bias-flat",
        "expression": "1.0 + x*x",
        "inputs": [
            {"name": "x", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-bias-flat", "x", 0.05).json()
    assert body["sensitivity"]["coefficient"] == 0.0
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is True
    assert iv["upper"] is None and iv["lower"] == 0.0


def test_biased_model_evaluation_pipeline_still_works(client):
    # 自适应差分是共享的梯度路径：偏置模型的 GUM+MC 传播照常可用且结果正确。
    payload = {
        "model_version": "pb-bias-small-sens",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.0, "distribution": "normal"},
            {"name": "b", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 10000.0, "distribution": "normal"},
        ],
        "correlations": [],
        "n": 4000, "seed": 20261005, "idempotency_key": "pb-bias-eval-0001",
    }
    r = client.post("/api/v1/evaluations", json=payload)
    assert r.status_code == 201, r.text
    gum = r.json()["result"]["combined"]["gum"]
    assert abs(gum["value"] - 1.0) < 1e-9
    assert abs(gum["standard_uncertainty"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 定义域边界：中心差分扩步被截断时，域内单侧探针确认真实斜率
# ---------------------------------------------------------------------------

def _sqrt_boundary_model(version: str):
    return {
        "version": version,
        "expression": "1.0 + 1e-10*sqrt(x)",
        "inputs": [
            {"name": "x", "value": 1e-8, "unit": "dimensionless",
             "standard_uncertainty": 1e-9, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }


def test_domain_boundary_slope_recovered_closed_interval(client):
    # y = 1.0 + 1e-10*sqrt(x)，x0 = 1e-8、ux = 1e-9，目标 1e-7，真实斜率 5e-7。
    # 中心差分扩步在 x-h<0 处被定义域截断（h 到不了 h_cap），差分值 2h·g
    # 达不到偏置 1.0 的百倍舍入门槛 -> 旧实现按噪声门槛误判零斜率、upper=null。
    # 域内单侧探针放大到 h_cap 确认依赖后，斜率取边界内中心差分估计：
    # v(u) = (g·u)^2 <= 1e-14 => u ∈ [0, ~0.19]（理论 0.2）闭区间，当前值达标。
    _register(client, _sqrt_boundary_model("pb-sqrt-boundary"))
    r = _budget(client, "pb-sqrt-boundary", "x", 1e-7)
    assert r.status_code == 200, r.text
    body = r.json()
    # 灵敏度恢复到 5e-7 附近（边界内中心差分估计，相对误差 ~4%）
    assert 4.5e-7 < body["sensitivity"]["coefficient"] < 5.5e-7
    assert body["sensitivity"]["unit"] == "dimensionless/dimensionless"
    iv = body["interval"]
    assert iv["feasible"] is True
    assert iv["unbounded_upper"] is False
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True   # 独立输入零端点闭
    assert abs(iv["upper"] - 0.19216334179121) < 1e-9            # 理论 0.2
    assert iv["upper_inclusive"] is True
    # 当前 ux = 1e-9 => uc ≈ 5.2e-16 ≤ 1e-7，达标
    assert body["satisfied"] is True
    assert body["current_combined_standard_uncertainty"]["value"] < 1e-7


def test_domain_boundary_gradient_matches_gum_pipeline(client):
    # 反求与 GUM 传播共用同一梯度路径：同一模型的 GUM 合成标准不确定度
    # 必须等于反求响应中的当前合成标准不确定度。
    payload = {
        "model_version": "pb-sqrt-boundary",
        "inputs": _sqrt_boundary_model("pb-sqrt-boundary")["inputs"],
        "correlations": [],
        "n": 5000, "seed": 20261005, "idempotency_key": "pb-sqrt-eval-0001",
    }
    r = client.post("/api/v1/evaluations", json=payload)
    assert r.status_code == 201, r.text
    gum_uc = r.json()["result"]["combined"]["gum"]["standard_uncertainty"]
    body = _budget(client, "pb-sqrt-boundary", "x", 1e-7).json()
    assert body["current_combined_standard_uncertainty"]["value"] == gum_uc


def test_domain_boundary_true_zero_slope_still_unbounded(client):
    # 同一边界结构下的真零斜率（0.0*sqrt(x)）：单侧探针到 h_cap 差值恒为 0，
    # 零灵敏度裁决不变 -> 任意 ux 可行，无上界。
    model = {
        "version": "pb-sqrt-flat",
        "expression": "1.0 + 0.0*sqrt(x)",
        "inputs": [
            {"name": "x", "value": 1e-8, "unit": "dimensionless",
             "standard_uncertainty": 1e-9, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    _register(client, model)
    body = _budget(client, "pb-sqrt-flat", "x", 1e-7).json()
    assert body["sensitivity"]["coefficient"] == 0.0
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is True
    assert iv["upper"] is None and iv["upper_inclusive"] is False
    assert iv["lower"] == 0.0 and iv["lower_inclusive"] is True
    assert body["satisfied"] is True


def test_domain_boundary_tight_target_not_satisfied(client):
    # 同一边界模型，目标 1e-16 < 当前 uc ≈ 5.2e-16：上界随目标收缩
    # （u ≤ t/g ≈ 1.9e-10），当前值不达标。
    body = _budget(client, "pb-sqrt-boundary", "x", 1e-16).json()
    iv = body["interval"]
    assert iv["feasible"] is True and iv["unbounded_upper"] is False
    # 响应序列化保留 15 位小数：1.9216334179121e-10 -> 1.92163e-10
    assert abs(iv["upper"] - 1.92163e-10) < 1e-20
    assert body["satisfied"] is False


# ---------------------------------------------------------------------------
# 只读语义：不改模型、不写计算记录；既有功能不受影响
# ---------------------------------------------------------------------------

def test_reverse_solving_does_not_write_or_mutate(client):
    before = client.get("/api/v1/evaluations").json()["count"]
    for _ in range(3):
        r = _budget(client, "pb-rm1", "a", 1.0)
        assert r.status_code == 200
    after = client.get("/api/v1/evaluations").json()["count"]
    assert after == before
    # 模型记录原样可查，登记的标准不确定度未被改动
    model = client.get("/api/v1/models/pb-rm1").json()
    ins = {s["name"]: s for s in model["inputs"]}
    assert ins["a"]["standard_uncertainty"] == 0.1
    assert ins["b"]["standard_uncertainty"] == 2.0


def test_reverse_solving_is_repeatable_and_does_not_consume_idempotency(client):
    r1 = _budget(client, "pb-rm1", "a", 1.0).json()
    r2 = _budget(client, "pb-rm1", "a", 1.0).json()
    assert r1 == r2
    # 幂等键命名空间不受影响：反求之后仍可正常提交计算
    payload = {
        "model_version": "pb-rm1",
        "inputs": _sum_model("pb-rm1", 0.1, 2.0, -1.0)["inputs"],
        "correlations": [{"i": "a", "j": "b", "r": -1.0}],
        "n": 4000, "seed": 20261003, "idempotency_key": "pb-after-reverse-0001",
    }
    r = client.post("/api/v1/evaluations", json=payload)
    assert r.status_code == 201, r.text
    assert abs(r.json()["result"]["combined"]["gum"]["standard_uncertainty"] - 1.9) < 1e-8
    # 再算一次幂等命中
    assert client.post("/api/v1/evaluations", json=payload).status_code == 200
    # 查询接口仍可用
    assert client.get("/api/v1/idempotency/pb-after-reverse-0001").status_code == 200
