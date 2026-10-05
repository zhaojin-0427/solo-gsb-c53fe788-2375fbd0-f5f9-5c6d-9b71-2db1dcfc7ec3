"""端到端测试：通过 FastAPI TestClient 走完整 HTTP 路径。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import storage


@pytest.fixture(scope="module")
def client():
    storage.init_db()
    with TestClient(app) as c:
        yield c


MODEL = {
    "version": "ohm-v1",
    "name": "直流功率",
    "description": "P = V*I",
    "expression": "V * I",
    "inputs": [
        {"name": "V", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1, "distribution": "normal"},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"},
    ],
    "output_unit": "W",
    "correlations": [{"i": "V", "j": "I", "r": 0.5}],
}


def eval_payload(**overrides):
    p = {
        "model_version": "ohm-v1",
        "inputs": [
            {"name": "V", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"},
        ],
        "correlations": [{"i": "V", "j": "I", "r": 0.5}],
        "n": 10000,
        "seed": 42,
        "idempotency_key": "key-0001-ohm-power",
    }
    p.update(overrides)
    return p


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_model(client):
    r = client.post("/api/v1/models", json=MODEL)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == "ohm-v1"
    assert body["summary"]["output_dimensionality"] == "[length]^2·[mass]·[time]^-3"
    assert body["summary"]["variables"] == ["I", "V"]


def test_register_duplicate_rejected(client):
    r = client.post("/api/v1/models", json=MODEL)
    assert r.status_code == 409


def test_expression_injection_rejected(client):
    bad = dict(MODEL)
    bad["version"] = "evil"
    bad["expression"] = "__import__('os').system('echo pwned')"
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422
    assert "ExpressionError" in r.text


def test_dimension_mismatch_rejected(client):
    bad = dict(MODEL)
    bad["version"] = "dim-bad"
    bad["output_unit"] = "meter"
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422


def test_unknown_function_rejected(client):
    bad = dict(MODEL)
    bad["version"] = "fn-bad"
    bad["expression"] = "eval(V)"
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422


def test_evaluation_flow(client):
    r = client.post("/api/v1/evaluations", json=eval_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["cached"] is False
    gum = body["result"]["combined"]["gum"]
    assert abs(gum["value"] - 20.0) < 1e-9
    assert abs(gum["standard_uncertainty"] - (0.3464 ** 2) ** 0.5) < 1e-3
    assert abs(gum["coverage_factor"] - 1.959964) < 1e-4
    mc = body["result"]["combined"]["monte_carlo"]
    assert 19.0 < mc["mean"] < 21.0
    assert mc["coverage_interval"]["low"] < mc["mean"] < mc["coverage_interval"]["high"]
    assert body["result"]["convergence"]["converged"] is True
    return body["id"]


def test_idempotent_repeat(client):
    key = "key-0009-idempotent"
    r1 = client.post("/api/v1/evaluations", json=eval_payload(idempotency_key=key))
    r2 = client.post("/api/v1/evaluations", json=eval_payload(idempotency_key=key))
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r1.json()["id"] == r2.json()["id"]


def test_idempotency_key_collision(client):
    p = eval_payload(n=5000)
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 409


def test_query_interfaces(client):
    listing = client.get("/api/v1/evaluations?model_version=ohm-v1").json()
    assert listing["count"] >= 1
    by_key = client.get("/api/v1/idempotency/key-0001-ohm-power")
    assert by_key.status_code == 200
    eid = by_key.json()["id"]
    ids = {item["id"] for item in listing["items"]}
    assert eid in ids
    one = client.get(f"/api/v1/evaluations/{eid}")
    assert one.status_code == 200
    assert one.json()["id"] == eid


def test_snapshot_mismatch_rejected(client):
    p = eval_payload(idempotency_key="key-0002-mismatch")
    p["inputs"][0]["value"] = 99.0
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 422
    assert "名义值" in r.json()["message"]


def test_unit_conversion_in_snapshot(client):
    # 10 V == 10000 mV；2 A == 2000 mA；等价单位写法必须被接受
    p = eval_payload(idempotency_key="key-0003-units")
    p["inputs"] = [
        {"name": "V", "value": 10000.0, "unit": "mV", "standard_uncertainty": 100.0, "distribution": "normal"},
        {"name": "I", "value": 2000.0, "unit": "mA", "standard_uncertainty": 20.0, "distribution": "normal"},
    ]
    p["correlations"] = [{"i": "V", "j": "I", "r": 0.5}]
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    assert abs(r.json()["result"]["combined"]["gum"]["value"] - 20.0) < 1e-9


def test_unit_dimension_incompatible_rejected(client):
    p = eval_payload(idempotency_key="key-0004-badunit")
    p["inputs"][0]["unit"] = "ampere"
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 422


def test_non_psd_correlation_rejected_and_not_saved(client):
    bad = dict(MODEL)
    bad["version"] = "ohm-3var"
    bad["inputs"] = [
        {"name": "V", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1, "distribution": "normal"},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"},
        {"name": "R", "value": 5.0, "unit": "ohm", "standard_uncertainty": 0.01, "distribution": "normal"},
    ]
    bad["expression"] = "V * I + (R - R) * (V * I) / R"
    bad["correlations"] = [
        {"i": "V", "j": "I", "r": 0.9},
        {"i": "I", "j": "R", "r": 0.9},
        {"i": "V", "j": "R", "r": -0.9},
    ]
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422
    assert "半正定" in r.json()["message"]
    # 模型没有被保存
    assert client.get("/api/v1/models/ohm-3var").status_code == 404


def test_zero_uncertainty_correlation_rejected(client):
    bad = dict(MODEL)
    bad["version"] = "zero-corr"
    bad["inputs"] = [
        {"name": "V", "value": 10.0, "unit": "V", "standard_uncertainty": 0.0, "distribution": "normal"},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"},
    ]
    bad["correlations"] = [{"i": "V", "j": "I", "r": 0.3}]
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422


def test_n_out_of_range(client):
    r = client.post("/api/v1/evaluations", json=eval_payload(n=999, idempotency_key="key-bad-n"))
    assert r.status_code == 422
    r = client.post("/api/v1/evaluations", json=eval_payload(n=100001, idempotency_key="key-bad-n2"))
    assert r.status_code == 422


def test_distributions_and_convergence_fields(client):
    body = dict(MODEL)
    body["version"] = "ohm-unif-tri"
    body["inputs"] = [
        {"name": "V", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1, "distribution": "uniform"},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02, "distribution": "triangular"},
    ]
    body["correlations"] = []
    r = client.post("/api/v1/models", json=body)
    assert r.status_code == 201, r.text

    payload = {
        "model_version": "ohm-unif-tri",
        "inputs": body["inputs"],
        "correlations": [],
        "n": 4000,
        "seed": 7,
        "idempotency_key": "key-0005-distributions",
    }
    r = client.post("/api/v1/evaluations", json=payload)
    assert r.status_code == 201, r.text
    conv = r.json()["result"]["convergence"]
    for key in ("se_mean", "se_std", "se_quantile_low", "se_quantile_high",
                "rel_se_mean", "rel_se_std", "rel_se_quantile", "kurtosis_excess"):
        assert key in conv
    contrib = r.json()["result"]["contributions"]["monte_carlo"]
    assert len(contrib) == 2


def test_nonconvergence_rejected_and_not_saved(client):
    # 极小 N（仍在允许区间）+ 极严格容差 => 不收敛 => 422，且结果不落库
    p = eval_payload(
        n=1000,
        convergence_tolerance=0.0001,
        idempotency_key="key-0006-noconverge",
    )
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 422
    assert "ConvergenceError" in r.text
    listing = client.get("/api/v1/evaluations?model_version=ohm-v1").json()
    ids = {item["idempotency_key"] for item in listing["items"]}
    assert "key-0006-noconverge" not in ids
    assert client.get("/api/v1/idempotency/key-0006-noconverge").status_code == 404


def test_reproducibility_same_seed(client):
    # 不同幂等键、相同参数 => 不同记录，但结果位级一致（固定种子）
    p1 = eval_payload(idempotency_key="key-0007-repro-a")
    p2 = eval_payload(idempotency_key="key-0007-repro-b")
    r1 = client.post("/api/v1/evaluations", json=p1).json()
    r2 = client.post("/api/v1/evaluations", json=p2).json()
    assert r1["result"]["combined"] == r2["result"]["combined"]


def test_domain_negative_log_rejected(client):
    # sqrt(a) 且 a 为正态、均值 1、标准差 10 => 抽样越界产生 NaN => 422 且不落库
    m = {
        "version": "sqrt-domain",
        "expression": "sqrt(a)",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 10.0, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }
    r = client.post("/api/v1/models", json=m)
    assert r.status_code == 201, r.text
    p = {
        "model_version": "sqrt-domain",
        "inputs": m["inputs"],
        "correlations": [],
        "n": 2000,
        "seed": 1,
        "idempotency_key": "key-0008-domain-nan",
    }
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 422
    assert "非有限" in r.json()["message"]


# ---------------------------------------------------------------------------
# 奇异半正定相关矩阵（r 恰为 ±1，两个标准不确定度均 > 0）
# ---------------------------------------------------------------------------

def _sum_model(version: str, r: float) -> dict:
    return {
        "version": version,
        "name": f"线性和 a+b，r={r}",
        "expression": "a + b",
        "inputs": [
            {"name": "a", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 5.0, "unit": "dimensionless",
             "standard_uncertainty": 0.2, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [{"i": "a", "j": "b", "r": r}],
    }


def _sum_payload(version: str, r: float, key: str, n: int = 10000, seed: int = 1234) -> dict:
    return {
        "model_version": version,
        "inputs": _sum_model(version, r)["inputs"],
        "correlations": [{"i": "a", "j": "b", "r": r}],
        "n": n,
        "seed": seed,
        "idempotency_key": key,
    }


def test_singular_correlation_models_register(client):
    # 修复前：r=±1 触发 Cholesky 失败 → 422；修复后：必须登记成功
    for version, r in (("sum-r-plus-v1", 1.0), ("sum-r-minus-v1", -1.0)):
        r_resp = client.post("/api/v1/models", json=_sum_model(version, r))
        assert r_resp.status_code == 201, r_resp.text
        assert client.get(f"/api/v1/models/{version}").status_code == 200


def test_perfect_positive_correlation_evaluation_and_query(client):
    # r=+1：GUM uc = ua + ub = 0.3；MC 同向，标准差收敛到 0.3
    p = _sum_payload("sum-r-plus-v1", 1.0, "key-0010-singular-plus")
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    body = r.json()
    gum = body["result"]["combined"]["gum"]
    assert abs(gum["value"] - 8.0) < 1e-9
    assert abs(gum["standard_uncertainty"] - 0.3) < 1e-9
    mc = body["result"]["combined"]["monte_carlo"]
    assert abs(mc["standard_uncertainty"] - 0.3) < 1e-2
    assert body["result"]["convergence"]["converged"] is True
    eval_id = body["id"]

    # 查询接口可用
    one = client.get(f"/api/v1/evaluations/{eval_id}")
    assert one.status_code == 200
    assert one.json()["id"] == eval_id
    by_key = client.get("/api/v1/idempotency/key-0010-singular-plus")
    assert by_key.status_code == 200
    assert by_key.json()["id"] == eval_id
    listing = client.get("/api/v1/evaluations?model_version=sum-r-plus-v1")
    assert listing.json()["count"] >= 1


def test_perfect_negative_correlation_evaluation(client):
    # r=-1：GUM uc = |ua - ub| = 0.1
    p = _sum_payload("sum-r-minus-v1", -1.0, "key-0011-singular-minus")
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    body = r.json()
    gum = body["result"]["combined"]["gum"]
    assert abs(gum["value"] - 8.0) < 1e-9
    assert abs(gum["standard_uncertainty"] - 0.1) < 1e-9

    # 幂等重放仍可用（200 + cached）
    r2 = client.post("/api/v1/evaluations", json=p)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r2.json()["id"] == body["id"]


def test_equal_uncertainty_anticorrelated_cancels(client):
    # ua == ub、r=-1：完全反向抵消，GUM uc = 0，覆盖区间退化为点；仍可登记/计算/查询
    m = {
        "version": "sum-cancel-v1",
        "expression": "a + b",
        "inputs": [
            {"name": "a", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 5.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [{"i": "a", "j": "b", "r": -1.0}],
    }
    r = client.post("/api/v1/models", json=m)
    assert r.status_code == 201, r.text
    p = {
        "model_version": "sum-cancel-v1",
        "inputs": m["inputs"],
        "correlations": m["correlations"],
        "n": 5000,
        "seed": 99,
        "idempotency_key": "key-0012-singular-cancel",
    }
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    gum = r.json()["result"]["combined"]["gum"]
    assert gum["standard_uncertainty"] == 0.0
    assert gum["coverage_interval"]["low"] == gum["coverage_interval"]["high"] == 8.0
    assert client.get(
        "/api/v1/idempotency/key-0012-singular-cancel"
    ).status_code == 200


def test_roundoff_negative_eigenvalue_registered_with_unit_diagonal(client):
    # 最小特征值处于 [-1e-8, 0)：修复为 PSD 且保持单位对角后放行
    m = {
        "version": "near-singular-v1",
        "expression": "a + b + c",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 2.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "c", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [
            {"i": "a", "j": "b", "r": 1.0 - 1e-11},
            {"i": "a", "j": "c", "r": 1.0 - 1e-11},
            {"i": "b", "j": "c", "r": 0.9999999995},
        ],
    }
    r = client.post("/api/v1/models", json=m)
    assert r.status_code == 201, r.text
    p = {
        "model_version": "near-singular-v1",
        "inputs": m["inputs"],
        "correlations": m["correlations"],
        "n": 4000,
        "seed": 5,
        "idempotency_key": "key-0013-near-singular",
    }
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text


def test_below_threshold_still_rejected(client):
    # 最小特征值低于 -1e-8 的矩阵仍须 422 拒绝，且不登记
    m = {
        "version": "below-threshold-v1",
        "expression": "a + b + c",
        "inputs": [
            {"name": "a", "value": 1.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 2.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "c", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [
            {"i": "a", "j": "b", "r": 1.0},
            {"i": "b", "j": "c", "r": 1.0},
            {"i": "a", "j": "c", "r": -1.0},
        ],
    }
    r = client.post("/api/v1/models", json=m)
    assert r.status_code == 422
    assert "半正定" in r.json()["message"]
    assert client.get("/api/v1/models/below-threshold-v1").status_code == 404


# ---------------------------------------------------------------------------
# 小量级标准不确定度（1e-9）：真实 ~1e-18 方差不得被固定绝对阈值钳零
# ---------------------------------------------------------------------------

def _tiny_model(version: str, ua: float, ub: float, r: float) -> dict:
    return {
        "version": version,
        "name": f"线性和 a+b（小量级，r={r}）",
        "expression": "a + b",
        "inputs": [
            {"name": "a", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": ua, "distribution": "normal"},
            {"name": "b", "value": 5.0, "unit": "dimensionless",
             "standard_uncertainty": ub, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [{"i": "a", "j": "b", "r": r}],
    }


def _tiny_payload(version: str, ua: float, ub: float, r: float, key: str) -> dict:
    return {
        "model_version": version,
        "inputs": _tiny_model(version, ua, ub, r)["inputs"],
        "correlations": [{"i": "a", "j": "b", "r": r}],
        "n": 20000,
        "seed": 20261003,
        "idempotency_key": key,
    }


def test_tiny_equal_uncertainty_perfect_positive_not_zeroed(client):
    # ua=ub=1e-9, r=+1 -> uc = 2e-9：标准不确定度、覆盖区间、逐项贡献均不得归零
    m = _tiny_model("sum-tiny-r1", 1e-9, 1e-9, 1.0)
    assert client.post("/api/v1/models", json=m).status_code == 201
    p = _tiny_payload("sum-tiny-r1", 1e-9, 1e-9, 1.0, "key-0014-tiny-plus")
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    body = r.json()
    gum = body["result"]["combined"]["gum"]
    assert abs(gum["value"] - 8.0) < 1e-12
    assert abs(gum["standard_uncertainty"] - 2e-9) < 1e-12
    ci = gum["coverage_interval"]
    assert ci["low"] < 8.0 < ci["high"]
    assert abs((ci["high"] - ci["low"]) / 2 - 1.959963984540054 * 2e-9) < 1e-14
    contrib = {c["source"]: c for c in body["result"]["contributions"]["gum"]}
    assert abs(contrib["a"]["variance_contribution"] - 1e-18) < 1e-24
    assert abs(contrib["b"]["variance_contribution"] - 1e-18) < 1e-24
    assert abs(contrib["a~b"]["variance_contribution"] - 2e-18) < 1e-24
    mc = body["result"]["combined"]["monte_carlo"]
    assert mc["standard_uncertainty"] > 0.0
    assert mc["coverage_interval"]["low"] < 8.0 < mc["coverage_interval"]["high"]
    assert body["result"]["convergence"]["converged"] is True

    # 幂等重放
    r2 = client.post("/api/v1/evaluations", json=p)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r2.json()["id"] == body["id"]

    # 结果查询
    assert client.get(f"/api/v1/evaluations/{body['id']}").status_code == 200
    assert client.get("/api/v1/idempotency/key-0014-tiny-plus").status_code == 200


def test_tiny_unequal_uncertainty_perfect_negative_not_zeroed(client):
    # ua=1e-9, ub=2e-9, r=-1 -> uc = 1e-9（不得归零）
    m = _tiny_model("sum-tiny-rm1", 1e-9, 2e-9, -1.0)
    assert client.post("/api/v1/models", json=m).status_code == 201
    p = _tiny_payload("sum-tiny-rm1", 1e-9, 2e-9, -1.0, "key-0015-tiny-minus")
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    gum = r.json()["result"]["combined"]["gum"]
    assert abs(gum["standard_uncertainty"] - 1e-9) < 1e-12
    assert gum["coverage_interval"]["low"] < 8.0 < gum["coverage_interval"]["high"]
    contrib = {c["source"]: c for c in r.json()["result"]["contributions"]["gum"]}
    assert contrib  # 逐项方差贡献仍然给出
    assert abs(contrib["a~b"]["variance_contribution"] - -4e-18) < 1e-24


def test_tiny_equal_uncertainty_perfect_negative_cancels(client):
    # ua=ub=1e-9, r=-1 -> 真正的完全抵消，uc = 0、覆盖区间退化为点
    m = _tiny_model("sum-tiny-cancel", 1e-9, 1e-9, -1.0)
    assert client.post("/api/v1/models", json=m).status_code == 201
    p = _tiny_payload("sum-tiny-cancel", 1e-9, 1e-9, -1.0, "key-0016-tiny-cancel")
    r = client.post("/api/v1/evaluations", json=p)
    assert r.status_code == 201, r.text
    gum = r.json()["result"]["combined"]["gum"]
    assert gum["standard_uncertainty"] == 0.0
    assert gum["coverage_interval"]["low"] == gum["coverage_interval"]["high"] == 8.0
    mc = r.json()["result"]["combined"]["monte_carlo"]
    assert mc["standard_uncertainty"] == 0.0
    assert mc["samples_range"] == {"min": 8.0, "max": 8.0}


def test_tiny_invalid_correlation_still_rejected_and_not_saved(client):
    # 小量级不改变非法相关矩阵的拒绝策略：422 且不保存
    bad = _tiny_model("sum-tiny-bad", 1e-9, 2e-9, 0.5)
    bad["inputs"].append({
        "name": "c", "value": 1.0, "unit": "dimensionless",
        "standard_uncertainty": 1e-9, "distribution": "normal",
    })
    bad["expression"] = "a + b + c"
    bad["correlations"] = [
        {"i": "a", "j": "b", "r": 1.0},
        {"i": "b", "j": "c", "r": 1.0},
        {"i": "a", "j": "c", "r": -1.0},
    ]
    r = client.post("/api/v1/models", json=bad)
    assert r.status_code == 422
    assert client.get("/api/v1/models/sum-tiny-bad").status_code == 404
