#!/usr/bin/env bash
# 最小端到端示例：登记 P=U*I 模型 -> 提交计算 -> 幂等重放 -> 查询
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

echo "== 健康检查 =="
curl -fsS "$BASE/health" && echo

echo "== 1. 登记模型 power-dc-v1（已存在则忽略 409）=="
curl -sS -X POST "$BASE/api/v1/models" -H 'Content-Type: application/json' -d '{
  "version": "power-dc-v1",
  "name": "直流功率 P = U*I",
  "expression": "U * I",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "output_unit": "W",
  "correlations": [{"i": "U", "j": "I", "r": 0.5}]
}' -o /tmp/model_resp.json -w "HTTP %{http_code}\n" || true

echo "== 2. 提交计算（N=20000，固定种子）=="
curl -sS -X POST "$BASE/api/v1/evaluations" -H 'Content-Type: application/json' -d '{
  "model_version": "power-dc-v1",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "correlations": [{"i": "U", "j": "I", "r": 0.5}],
  "n": 20000,
  "seed": 20260929,
  "idempotency_key": "demo-power-0001"
}' -o /tmp/eval_resp.json -w "HTTP %{http_code}\n"

python3 - <<'PY'
import json
d = json.load(open('/tmp/eval_resp.json'))
g, m = d["result"]["combined"]["gum"], d["result"]["combined"]["monte_carlo"]
print(f"GUM : {g['value']:.4f} ± {g['standard_uncertainty']:.4f} {g['unit']} "
      f"95%=[{g['coverage_interval']['low']:.4f}, {g['coverage_interval']['high']:.4f}]")
print(f"MC  : {m['mean']:.4f} ± {m['standard_uncertainty']:.4f} {m['unit']} "
      f"95%=[{m['coverage_interval']['low']:.4f}, {m['coverage_interval']['high']:.4f}]")
print("evaluation id:", d["id"], "| cached:", d.get("cached"))
PY

echo "== 3. 幂等重放（期望 HTTP 200, cached=true）=="
curl -sS -X POST "$BASE/api/v1/evaluations" -H 'Content-Type: application/json' -d '{
  "model_version": "power-dc-v1",
  "inputs": [
    {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
    {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
  ],
  "correlations": [{"i": "U", "j": "I", "r": 0.5}],
  "n": 20000,
  "seed": 20260929,
  "idempotency_key": "demo-power-0001"
}' -o /dev/null -w "HTTP %{http_code}\n"

echo "== 4. 查询 =="
curl -fsS "$BASE/api/v1/idempotency/demo-power-0001" -o /dev/null -w "by key: HTTP %{http_code}\n"
curl -fsS "$BASE/api/v1/models/power-dc-v1" -o /dev/null -w "model  : HTTP %{http_code}\n"

echo "== 5. 精度反求（POST /api/v1/precision-budgets，只读）=="
# 登记负相关线性模型 y=a+b（r=-1, ub=2）；反求目标 uc(y)<=1 时 ua 的可行区间，理论 [1,3]
curl -sS -X POST "$BASE/api/v1/models" -H 'Content-Type: application/json' -d '{
  "version": "budget-sum-rm1",
  "name": "线性和 y=a+b（负相关反求示例）",
  "expression": "a + b",
  "inputs": [
    {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 0.1, "distribution": "normal"},
    {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 2.0, "distribution": "normal"}
  ],
  "output_unit": "dimensionless",
  "correlations": [{"i": "a", "j": "b", "r": -1.0}]
}' -o /dev/null -w "model  : HTTP %{http_code}\n" || true

curl -sS -X POST "$BASE/api/v1/precision-budgets" -H 'Content-Type: application/json' -d '{
  "model_version": "budget-sum-rm1",
  "input_name": "a",
  "target": {"standard_uncertainty": 1.0, "unit": "dimensionless"}
}' -o /tmp/budget_resp.json -w "budget : HTTP %{http_code}\n"

python3 - <<'PY'
import json
d = json.load(open('/tmp/budget_resp.json'))
iv = d["interval"]
if iv["feasible"]:
    lo = ("[" if iv["lower_inclusive"] else "(") + str(iv["lower"])
    hi = ("∞" if iv["upper"] is None else str(iv["upper"])) + ("]" if iv["upper_inclusive"] else ")")
    print(f"可行 ua 区间: {lo}, {hi}  （理论 [1, 3]，负相关双侧有界）")
else:
    print("可行区间为空（目标无法达到）")
print("当前登记值是否达标:", d["satisfied"],
      "| 当前 uc(y) =", d["current_combined_standard_uncertainty"]["value"])
PY

echo "== 6. 反求不写计算记录（计数不因第 5 步增加）=="
curl -fsS "$BASE/api/v1/evaluations?model_version=budget-sum-rm1" \
  | python3 -c 'import json,sys; print("budget-sum-rm1 的计算记录数:", json.load(sys.stdin)["count"])'

echo "== 7. 偏置模型小灵敏度反求（y=a+1e-4*b，自适应步长，上界约 5000）=="
curl -sS -X POST "$BASE/api/v1/models" -H 'Content-Type: application/json' -d '{
  "version": "budget-bias",
  "name": "带偏置的小灵敏度 y=a+1e-4*b",
  "expression": "a + 1e-4*b",
  "inputs": [
    {"name": "a", "value": 1.0, "unit": "dimensionless", "standard_uncertainty": 0.0, "distribution": "normal"},
    {"name": "b", "value": 0.0, "unit": "dimensionless", "standard_uncertainty": 10000.0, "distribution": "normal"}
  ],
  "output_unit": "dimensionless"
}' -o /dev/null -w "model  : HTTP %{http_code}\n" || true

curl -sS -X POST "$BASE/api/v1/precision-budgets" -H 'Content-Type: application/json' -d '{
  "model_version": "budget-bias",
  "input_name": "b",
  "target": {"standard_uncertainty": 0.5, "unit": "dimensionless"}
}' -o /tmp/budget_bias_resp.json -w "budget : HTTP %{http_code}\n"

python3 - <<'PY'
import json
d = json.load(open('/tmp/budget_bias_resp.json'))
iv = d["interval"]
print("灵敏度:", d["sensitivity"]["coefficient"], "（真实 1e-4，不得被判零）")
if iv["unbounded_upper"]:
    print("错误：有限预算被判为无上界")
else:
    print(f"可行 ub 区间: [{iv['lower']}, {iv['upper']}]  （理论 [0, 5000]，闭区间）")
print("当前登记值是否达标:", d["satisfied"],
      "| 当前 uc(y) =", d["current_combined_standard_uncertainty"]["value"])
PY

echo "== 8. 近定义域边界反求（y=1+1e-10*sqrt(x)，x0=1e-8，上界约 0.2）=="
# 中心差分扩步被 sqrt 定义域截断（x-h<0）；域内单侧探针确认真实斜率 5e-7
curl -sS -X POST "$BASE/api/v1/models" -H 'Content-Type: application/json' -d '{
  "version": "budget-sqrt-boundary",
  "name": "近定义域边界的小斜率 y=1+1e-10*sqrt(x)",
  "expression": "1.0 + 1e-10*sqrt(x)",
  "inputs": [
    {"name": "x", "value": 1e-8, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"}
  ],
  "output_unit": "dimensionless"
}' -o /dev/null -w "model  : HTTP %{http_code}\n" || true

curl -sS -X POST "$BASE/api/v1/precision-budgets" -H 'Content-Type: application/json' -d '{
  "model_version": "budget-sqrt-boundary",
  "input_name": "x",
  "target": {"standard_uncertainty": 1e-7, "unit": "dimensionless"}
}' -o /tmp/budget_sqrt_resp.json -w "budget : HTTP %{http_code}\n"

python3 - <<'PY'
import json
d = json.load(open('/tmp/budget_sqrt_resp.json'))
iv = d["interval"]
print("灵敏度:", d["sensitivity"]["coefficient"], "（真实 5e-7，边界截断不得判零）")
if iv["unbounded_upper"]:
    print("错误：有限预算被判为无上界")
else:
    lo = "[" if iv["lower_inclusive"] else "("
    hi = "]" if iv["upper_inclusive"] else ")"
    print(f"可行 ux 区间: {lo}{iv['lower']}, {iv['upper']}{hi}  （理论 [0, 0.2]，闭区间）")
print("当前登记值是否达标:", d["satisfied"],
      "| 当前 uc(y) =", d["current_combined_standard_uncertainty"]["value"])
PY
