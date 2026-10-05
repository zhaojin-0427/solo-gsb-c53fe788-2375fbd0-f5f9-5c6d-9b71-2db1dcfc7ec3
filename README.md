# 实验测量不确定度传播 API

基于 **Python · FastAPI · SQLite · NumPy · Pint** 实现的实验测量不确定度传播服务。

客户端登记带单位的输入量（名义值、标准不确定度、分布、两两相关系数）与**不可变版本化测量模型**（表达式仅允许白名单算子/函数）；提交计算请求后，服务依次执行：

1. **单位换算与维度校验**（Pint）——输入快照换算到模型声明单位，推导表达式量纲并与输出单位核对；
2. **相关矩阵校验**——对称、对角为 1、|r|≤1、半正定（**允许奇异**：两个标准不确定度均大于零的输入相关系数恰为 ±1 时，矩阵含合法零特征值，直接接受；最小特征值低于 -1e-8 判为明显非半正定并拒绝；处于 [-1e-8, 0) 的舍入误差按 Higham 最近相关矩阵算法修复为半正定且保持单位对角）；
3. **GUM 一阶线性化**（JCGM 100:2008）——合成标准不确定度、95% 覆盖区间、解析灵敏度系数与逐输入方差贡献；
4. **固定种子蒙特卡洛**（N ∈ [1000, 100000]，Gaussian copula 保证相关结构与边际分布同时精确）——合成值、2.5%/97.5% 分位覆盖区间、逐输入相关贡献与收敛统计。

**非法相关矩阵、量纲不兼容、抽样越界（NaN/Inf）或蒙特卡洛不收敛一律返回 422 且不保存任何部分结果。**

请求按 `模型版本 + 输入快照 + N + 种子 + 收敛容差 + 幂等键` 绑定，重复提交只返回同一结果。

此外提供只读的 **测量精度反求接口** `POST /api/v1/precision-budgets`：给定已登记模型、单个待调整输入与带单位的非负目标 GUM 合成标准不确定度，在不改模型、不写计算记录的前提下，反解该输入标准不确定度（含全部协方差/负相关项）的可行区间，详见 [第 9 节](#9-测量精度反求precision-budgets)。

---

## 目录

- [快速开始（Docker Compose 一键启动）](#快速开始docker-compose-一键启动)
- [本地开发运行](#本地开发运行)
- [接口总览](#接口总览)
- [完整示例](#完整示例)
  - [1. 登记测量模型](#1-登记测量模型)
  - [2. 提交计算请求](#2-提交计算请求)
  - [3. 重复请求（幂等）](#3-重复请求幂等)
  - [4. 查询结果](#4-查询结果)
  - [5. 等价单位写法](#5-等价单位写法)
  - [6. 拒绝场景示例](#6-拒绝场景示例)
  - [7. 完全相关（r = ±1，奇异半正定矩阵）](#7-完全相关r--1奇异半正定矩阵)
  - [8. 小量级标准不确定度（1e-9，真实方差不得归零）](#8-小量级标准不确定度1e-9真实方差不得归零)
  - [9. 测量精度反求（precision-budgets）](#9-测量精度反求precision-budgets)
- [模型表达式白名单](#模型表达式白名单)
- [分布与相关系数](#分布与相关系数)
- [结果字段说明](#结果字段说明)
- [幂等与“不保存部分结果”语义](#幂等与不保存部分结果语义)
- [运行测试](#运行测试)
- [项目结构](#项目结构)

---

## 快速开始（Docker Compose 一键启动）

前置条件：已安装 Docker 与 Docker Compose 插件。

```bash
docker compose up --build -d
```

服务监听在 <http://localhost:8000>：

| 地址 | 说明 |
| --- | --- |
| <http://localhost:8000/> | 服务信息 |
| <http://localhost:8000/health> | 健康检查 |
| <http://localhost:8000/docs> | Swagger UI（可直接在浏览器里发请求） |
| <http://localhost:8000/redoc> | ReDoc 文档 |

SQLite 数据库保存在命名卷 `uncertainty-data`（容器内 `/data/uncertainty.db`），容器重建后数据仍在。

全部接口随该命令一同启动（含只读的测量精度反求接口 `POST /api/v1/precision-budgets`，无需额外配置）；反求接口的调用示例见 [第 9 节](#9-测量精度反求precision-budgets)。

停止：

```bash
docker compose down          # 保留数据
docker compose down -v       # 连数据卷一起删除
```

## 本地开发运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 数据库路径可通过环境变量覆盖（默认 /data/uncertainty.db）
export UNCERTAINTY_DB_PATH=./data/uncertainty.db
python run.py                 # 或 uvicorn app.main:app --reload --port 8000
```

## 接口总览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/models` | 登记不可变版本化模型（版本重复 → 409） |
| GET | `/api/v1/models/{version}` | 查询模型（含规范化模型摘要） |
| GET | `/api/v1/models?limit=&offset=` | 列出模型 |
| POST | `/api/v1/evaluations` | 提交计算（首次 201；幂等命中 200 且 `cached=true`） |
| GET | `/api/v1/evaluations/{id}` | 按结果 ID 查询 |
| GET | `/api/v1/evaluations?model_version=` | 列出计算结果（可按模型版本过滤） |
| GET | `/api/v1/idempotency/{key}` | 按幂等键查询 |
| POST | `/api/v1/precision-budgets` | 测量精度反求：反解单个输入标准不确定度的可行区间（只读，200） |
| GET | `/health` | 健康检查 |

## 完整示例

以直流功率 `P = U·I` 为例：U = 10.0(10) V，I = 2.00(2) A（括号内为标准不确定度），二者相关系数 0.5，均为正态分布。

### 1. 登记测量模型

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "power-dc-v1",
    "name": "直流功率 P = U*I",
    "description": "由电压、电流测量值计算直流功率（相关输入示例）",
    "expression": "U * I",
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
      {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
    ],
    "output_unit": "W",
    "correlations": [{"i": "U", "j": "I", "r": 0.5}]
  }'
```

返回 `201`，其中 `summary` 为**规范化模型摘要**：AST 规范化表达式及其 SHA-256、按名排序的变量列表、每个输入的单位与量纲、输出量纲、相关系数与白名单清单：

```json
{
  "version": "power-dc-v1",
  "expression": "U * I",
  "expression_canonical": "U * I",
  "output_unit": "W",
  "summary": {
    "expression": "U * I",
    "expression_sha256": "aa939c0d0fb9cf83a06670aef9deecd0eaf405feb07af2d0eb94c4abe60fce8a",
    "variables": ["I", "U"],
    "inputs": [
      {"name": "I", "unit": "A", "dimensionality": "[current]", "distribution": "normal"},
      {"name": "U", "unit": "V", "dimensionality": "[current]^-1·[length]^2·[mass]·[time]^-3", "distribution": "normal"}
    ],
    "output_unit": "W",
    "output_dimensionality": "[length]^2·[mass]·[time]^-3",
    "correlations": [["I", "U", 0.5]],
    "whitelist": { "operators": ["+", "-", "*", "/", "**", "unary+", "unary-"], "...": "..." }
  }
}
```

### 2. 提交计算请求

```bash
curl -sS -X POST http://localhost:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "power-dc-v1",
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,  "distribution": "normal"},
      {"name": "I", "value": 2.0,  "unit": "A", "standard_uncertainty": 0.02, "distribution": "normal"}
    ],
    "correlations": [{"i": "U", "j": "I", "r": 0.5}],
    "n": 20000,
    "seed": 20260929,
    "idempotency_key": "demo-power-0001"
  }'
```

返回 `201`（节选；数值随种子确定，可复现）：

```json
{
  "id": "2fe78121179b4241b4b11cc4acd1e52e",
  "request_hash": "dc83ec665ac4cd5dd944d1c4121aab6644d6f5cd8d06c4fc7ec6ebfdd8f3e037",
  "idempotency_key": "demo-power-0001",
  "result": {
    "combined": {
      "gum": {
        "value": 20.0,
        "standard_uncertainty": 0.346410161511773,
        "coverage_interval": {"low": 19.32104855955822, "high": 20.67895144044178},
        "coverage_factor": 1.959963984540054,
        "unit": "W"
      },
      "monte_carlo": {
        "mean": 20.003607269122142,
        "standard_uncertainty": 0.349635117087063,
        "median": 20.000662952454014,
        "coverage_interval": {"low": 19.320035649804154, "high": 20.687972138617937},
        "samples_range": {"min": 18.764296221280656, "max": 21.394827454449107},
        "unit": "W"
      }
    },
    "sensitivities": {
      "I": {"coefficient": 10.0, "unit": "W/A"},
      "U": {"coefficient": 2.0,  "unit": "W/V"}
    },
    "contributions": {
      "gum": [
        {"source": "U",     "kind": "individual",  "variance_contribution": 0.039999999999537, "fraction": 0.333333333333333},
        {"source": "I",     "kind": "individual",  "variance_contribution": 0.039999999999537, "fraction": 0.333333333333333},
        {"source": "U~I",   "kind": "covariance",  "variance_contribution": 0.039999999999537, "fraction": 0.333333333333333}
      ],
      "monte_carlo": [
        {"source": "U", "pearson_r": 0.866970679085141, "r_squared": 0.751638158393351},
        {"source": "I", "pearson_r": 0.868017294162284, "r_squared": 0.753454022964814}
      ]
    },
    "convergence": {
      "n": 20000, "seed": 20260929, "kurtosis_excess": 0.00061862452913,
      "se_mean": 0.002472293622332, "se_std": 0.001748445930606,
      "se_quantile_low": 0.00631453621075, "se_quantile_high": 0.006789317774467,
      "rel_se_mean": 0.000123592389566, "rel_se_std": 0.005000773220874, "rel_se_quantile": 0.004963182011728,
      "tolerance": 0.05, "converged": true
    }
  }
}
```

> 理论验证：u²(P) = (I·u_U)² + (U·u_I)² + 2UI·r·u_U·u_I = 0.04+0.04+0.04 = 0.12，u(P)=√0.12≈0.3464 W。

### 3. 重复请求（幂等）

原样再发一次同样的请求，返回 `200` 且：

```json
{ "cached": true, "id": "2fe78121179b4241b4b11cc4acd1e52e", "...": "结果完全相同" }
```

**相同幂等键但内容不同**（例如改 N）→ `409 Conflict`。

### 4. 查询结果

```bash
curl -sS http://localhost:8000/api/v1/evaluations/2fe78121179b4241b4b11cc4acd1e52e
curl -sS "http://localhost:8000/api/v1/evaluations?model_version=power-dc-v1"
curl -sS http://localhost:8000/api/v1/idempotency/demo-power-0001
```

### 5. 等价单位写法

输入快照允许使用量纲等价的任意 Pint 单位（自动换算后与登记模型核对）。10 V/0.1 V 写成 mV：

```json
"inputs": [
  {"name": "U", "value": 10000, "unit": "mV", "standard_uncertainty": 100, "distribution": "normal"},
  {"name": "I", "value": 2000,  "unit": "mA", "standard_uncertainty": 20,  "distribution": "normal"}
]
```

结果完全一致（P = 20.0000 W，u(P) = 0.3464 W）。若单位量纲不兼容（如给电压传 `ampere`）→ 422。

### 6. 拒绝场景示例

| 场景 | 状态码 | 错误示例 |
| --- | --- | --- |
| 表达式注入 / 非白名单结构（`__import__('os')`、属性访问、比较、lambda…） | 422 | `ExpressionError: 只允许调用白名单中的简单函数名` |
| 非半正定相关矩阵（最小特征值 < -1e-8） | 422 | `CorrelationError: 相关矩阵不是半正定矩阵：最小特征值 -8.000e-01 < -1e-08` |
| 近奇异矩阵（最小特征值 ∈ [-1e-8, 0)） | 201/200 | 按舍入误差修复为半正定并保持单位对角，登记/计算照常进行 |
| 完全相关 r = ±1（两个 u 均 > 0，矩阵奇异） | 201/200 | 零特征值合法；见[下节](#7-完全相关r--1奇异半正定矩阵) |
| 表达式量纲矛盾（`m + kg`）/ 输出单位量纲不符 | 422 | `RequestError: 模型维度不兼容 ...` |
| 快照与登记模型不一致（值/不确定度/分布/相关对） | 422 | `RequestError: 输入 'U' 的名义值与登记模型不符 ...` |
| 相关系数引用 u=0 的输入量 | 422 | `CorrelationError: ... 引用了标准不确定度为 0 的输入量` |
| N 越界（<1000 或 >100000） | 422 | Pydantic 校验错误 |
| 蒙特卡洛不收敛 | 422 | `ConvergenceError: 蒙特卡洛 N=1000 未通过收敛判据 ...` |
| 抽样越过模型定义域（如对正态量取 sqrt/log 产生 NaN） | 422 | `PropagationError: 蒙特卡洛抽样中模型产生了 k/N 个非有限结果 ...` |
| 幂等键绑定到不同请求 / 模型版本重复 | 409 | `ConflictError` |
| 精度反求：未知模型版本 | 404 | `NotFoundError: 模型版本 ... 不存在`（见 [9.4](#94-拒绝场景)） |
| 精度反求：未知输入名 / 目标量纲不符 / 目标负值或非有限 | 422 | `RequestError` / Pydantic 校验错误（见 [9.4](#94-拒绝场景)） |

### 7. 完全相关（r = ±1，奇异半正定矩阵）

两个输入的标准不确定度**均大于零**、相关系数恰为 **+1 或 −1** 时，相关矩阵的最小特征值为 **0**（奇异半正定，而非非半正定）。这是合法输入：登记返回 201，计算与查询照常进行。

以线性模型 `y = a + b`（a = 3.0(10)，b = 5.0(20)，均无量纲、正态）为例，GUM 合成标准不确定度为

```
u(y) = |ua + r·ub|
```

即 r = +1 时 u(y) = 0.30（同向相加），r = −1 时 u(y) = 0.10（反向相减）。固定种子蒙特卡洛中，两个标准化输入**严格同向（r=+1）或反向（r=−1）变化**（逐样本位级一致/相反，不是近似），因此 MC 标准差收敛到同一解析值。

**登记（r = +1）：**

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "sum-perfect-r1",
    "name": "线性和 y=a+b（完全正相关）",
    "expression": "a + b",
    "inputs": [
      {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 0.1, "distribution": "normal"},
      {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 0.2, "distribution": "normal"}
    ],
    "output_unit": "dimensionless",
    "correlations": [{"i": "a", "j": "b", "r": 1.0}]
  }'
# HTTP 201（奇异半正定矩阵被接受；summary.correlations = [["a","b",1.0]]）
```

**计算（N=20000，seed=20260930）：**

```bash
curl -sS -X POST http://localhost:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "sum-perfect-r1",
    "inputs": [
      {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 0.1, "distribution": "normal"},
      {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 0.2, "distribution": "normal"}
    ],
    "correlations": [{"i": "a", "j": "b", "r": 1.0}],
    "n": 20000, "seed": 20260930, "idempotency_key": "doc-r1-0002"
  }'
```

返回 `201`（以下为该种子下的**实际响应**节选，可复现）：

```json
{
  "id": "2cdf1a18514849f3afedf2a32d820d85",
  "cached": false,
  "result": {
    "combined": {
      "gum": {
        "value": 8.0,
        "standard_uncertainty": 0.300000000004133,
        "coverage_interval": {"low": 7.412010804629884, "high": 8.587989195370117},
        "coverage_factor": 1.959963984540054,
        "unit": "dimensionless"
      },
      "monte_carlo": {
        "mean": 7.999672567876607,
        "standard_uncertainty": 0.301707430632374,
        "median": 7.997800889847241,
        "coverage_interval": {"low": 7.410332065710037, "high": 8.59332288699073},
        "samples_range": {"min": 6.780157415611496, "max": 9.190815122599634},
        "unit": "dimensionless"
      }
    },
    "contributions": {
      "monte_carlo": [
        {"source": "a", "pearson_r": 1.0, "r_squared": 1.0},
        {"source": "b", "pearson_r": 1.0, "r_squared": 1.0}
      ]
    },
    "convergence": {"n": 20000, "seed": 20260930, "converged": true}
  }
}
```

> GUM 值 0.300000000004 与精确值 0.30 的 4e-12 偏差来自中心差分灵敏度系数（∂y/∂a 数值为 1.000000000024），与相关矩阵是否奇异无关；MC 标准差 0.3017 是 N=20000 的抽样涨落（相对标准误约 0.5%）。

把模型与请求中的 `r` 改成 **−1.0**（版本 `sum-perfect-rm1`、幂等键 `doc-rm1-0001`，其余不变），返回 `201`：

```json
{
  "id": "fd5034c82af94f2f8ef82302b52b7bc1",
  "result": {
    "combined": {
      "gum": {
        "value": 8.0,
        "standard_uncertainty": 0.099999999999422,
        "coverage_interval": {"low": 7.804003601547127, "high": 8.195996398452872},
        "unit": "dimensionless"
      },
      "monte_carlo": {
        "mean": 7.999887180315274,
        "standard_uncertainty": 0.100156418055492,
        "coverage_interval": {"low": 7.800771459463655, "high": 8.196261317942582},
        "samples_range": {"min": 7.628435460607403, "max": 8.392275772289075},
        "unit": "dimensionless"
      }
    },
    "contributions": {
      "monte_carlo": [
        {"source": "a", "pearson_r": -1.0, "r_squared": 1.0},
        {"source": "b", "pearson_r": 1.0,  "r_squared": 1.0}
      ]
    }
  }
}
```

**完全抵消边界**：r = −1 且 ua = ub（如都为 0.1）时，两个标准化正态严格反向、逐项抵消，输出退化为确定值。GUM 与 MC 均给出 `standard_uncertainty = 0.0`、覆盖区间 `[8.0, 8.0]`，MC 的 `samples_range` 也是 `{min: 8.0, max: 8.0}`，收敛判据照常通过（版本 `sum-perfect-cancel`，N=5000，seed=77，幂等键 `doc-cancel-0001`）。

查询接口对这些结果完全一致地可用：

```bash
curl -sS http://localhost:8000/api/v1/evaluations/2cdf1a18514849f3afedf2a32d820d85   # 200
curl -sS http://localhost:8000/api/v1/idempotency/doc-r1-0002                        # 200
curl -sS "http://localhost:8000/api/v1/evaluations?model_version=sum-perfect-r1"     # 200
```

**边界规则**（与实现一致）：

| 最小特征值 | 处置 |
| --- | --- |
| = 0（r 恰为 ±1 的完全相关） | 接受，矩阵原样保留，零特征值可用于抽样 |
| ∈ [−1e-8, 0)（舍入误差） | 投影到最近相关矩阵（半正定 + 保持单位对角）后接受 |
| < −1e-8（明显非半正定） | 422 拒绝，不登记、不保存 |

抽样实现上，正定矩阵仍走 Cholesky 分解（固定种子流与历史版本逐位一致）；奇异半正定矩阵改用特征分解的对称平方根 F = Q·√Λ·Qᵀ（F Fᵀ = 相关矩阵），并对矩阵中精确存储为 ±1 的边做符号对齐，因此 r=±1 时标准化正态行严格相同/相反。正态边际直接使用该标准正态样本（不经 CDF/分位往返），保证完全反相时严格抵消；均匀/三角边际仍经 Gaussian copula 变换保持精确边际分布。

### 8. 小量级标准不确定度（1e-9，真实方差不得归零）

判定 GUM 合成方差是否“舍入噪声”的阈值**必须与协方差项同尺度**：阈值取

```
ε · n · Σᵢⱼ |gᵢ gⱼ Cov(i,j)|
```

（ε 为机器精度，n 为输入个数），随不确定度量级一起收缩。绝不能用与输入尺度无关的固定绝对阈值（如 `ε·max(1, …)`）——当两个标准不确定度均为 **1e-9** 时，真实合成方差只有 ~1e-18 量级，固定阈值（~2.2e-16）会把它误判为舍入噪声而**钳为零**，导致标准不确定度、95% 覆盖区间与逐项方差贡献全部错误归零。响应序列化同样按尺度处理：常规数值保留历史的 15 位小数舍入，小于 1e-15 的小值改用 15 位有效数字，避免 1e-18 被舍成 0.0。

以线性模型 `y = a + b`（a = 3.0、b = 5.0，均无量纲、正态）为例：

| ua | ub | r | 理论 u(y) = |ua + r·ub| | 处置 |
| --- | --- | --- | --- | --- |
| 1e-9 | 1e-9 | +1 | **2e-9** | 真实正方差，完整保留 |
| 1e-9 | 2e-9 | −1 | **1e-9** | 真实正方差，完整保留 |
| 1e-9 | 1e-9 | −1 | **0** | 理论完全抵消，钳为零 |

**登记（ua = ub = 1e-9，r = +1）：**

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "sum-tiny-r1",
    "name": "线性和 y=a+b（小量级完全正相关）",
    "expression": "a + b",
    "inputs": [
      {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"},
      {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"}
    ],
    "output_unit": "dimensionless",
    "correlations": [{"i": "a", "j": "b", "r": 1.0}]
  }'
# HTTP 201
```

**计算（N=20000，seed=20261003）：**

```bash
curl -sS -X POST http://localhost:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "sum-tiny-r1",
    "inputs": [
      {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"},
      {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"}
    ],
    "correlations": [{"i": "a", "j": "b", "r": 1.0}],
    "n": 20000, "seed": 20261003, "idempotency_key": "doc-tiny-r1-0001"
  }'
```

返回 `201`（以下为该种子下的**实际响应**节选，可复现）：

```json
{
  "id": "a09a0c76418d425a90c8fccd532465dd",
  "cached": false,
  "result": {
    "combined": {
      "gum": {
        "value": 8.0,
        "standard_uncertainty": 2.00000000003244e-09,
        "coverage_interval": {"low": 7.999999996080072, "high": 8.000000003919927},
        "coverage_factor": 1.959963984540054,
        "unit": "dimensionless"
      },
      "monte_carlo": {
        "mean": 7.9999999999899,
        "standard_uncertainty": 2.0060324203627e-09,
        "median": 7.99999999997027,
        "coverage_interval": {"low": 7.999999996075437, "high": 8.000000003927587},
        "samples_range": {"min": 7.999999991104339, "max": 8.00000000732518},
        "unit": "dimensionless"
      }
    },
    "contributions": {
      "gum": [
        {"source": "a",   "kind": "individual", "variance_contribution": 1.00000000004711e-18, "fraction": 0.250000000003667},
        {"source": "b",   "kind": "individual", "variance_contribution": 1.00000000001777e-18, "fraction": 0.249999999996333},
        {"source": "a~b", "kind": "covariance", "variance_contribution": 2.00000000006488e-18, "fraction": 0.5}
      ],
      "monte_carlo": [
        {"source": "a", "pearson_r": 0.99999999999995,  "r_squared": 0.999999999999901},
        {"source": "b", "pearson_r": 0.999999999999902, "r_squared": 0.999999999999804}
      ]
    },
    "convergence": {"n": 20000, "seed": 20261003, "converged": true}
  }
}
```

> GUM 标准不确定度 2.00000000003e-9 ≈ 精确值 2e-9（偏差来自中心差分梯度 ~2e-11 相对误差）；95% 覆盖区间半宽 3.92e-9 = 1.95996·2e-9；逐项方差贡献 1e-18 / 1e-18 / 2e-18（比例 0.25 / 0.25 / 0.5）均**不为零**。

**r = −1、ua = 1e-9、ub = 2e-9（版本 `sum-tiny-rm1`，幂等键 `doc-tiny-rm1-0001`，同种子）**，返回 `201`：

```json
{
  "id": "7204d809af4f4aa484b781d47df9e348",
  "result": {
    "combined": {
      "gum": {
        "value": 8.0,
        "standard_uncertainty": 9.99999999994219e-10,
        "coverage_interval": {"low": 7.999999998040036, "high": 8.000000001959965},
        "unit": "dimensionless"
      },
      "monte_carlo": {
        "standard_uncertainty": 1.00492046676458e-09,
        "coverage_interval": {"low": 7.99999999803479, "high": 8.000000001971163},
        "samples_range": {"min": 7.999999996356614, "max": 8.000000004246077}
      }
    },
    "contributions": {
      "gum": [
        {"source": "a",   "kind": "individual", "variance_contribution": 1.00000000004711e-18,  "fraction": 1.000000000058668},
        {"source": "b",   "kind": "individual", "variance_contribution": 4.00000000007109e-18,  "fraction": 4.000000000117337},
        {"source": "a~b", "kind": "covariance", "variance_contribution": -4.00000000012976e-18, "fraction": -4.000000000176005}
      ]
    }
  }
}
```

**两者同为 1e-9、r = −1（版本 `sum-tiny-cancel`，N=5000，seed=77，幂等键 `doc-tiny-cancel-0001`）** 才是真正的完全抵消：GUM 与 MC 均给出 `standard_uncertainty = 0.0`、覆盖区间 `[8.0, 8.0]`，MC `samples_range` 也是 `{min: 8.0, max: 8.0}`，收敛判据照常通过。

幂等重放、结果查询与非法矩阵拒绝对小量级模型完全一致：

```bash
# 原样重放上面的 doc-tiny-r1-0001 请求 -> HTTP 200，body 中 cached=true，id 相同
curl -sS -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{"model_version":"sum-tiny-r1",
       "inputs":[{"name":"a","value":3.0,"unit":"dimensionless","standard_uncertainty":1e-9,"distribution":"normal"},
                 {"name":"b","value":5.0,"unit":"dimensionless","standard_uncertainty":1e-9,"distribution":"normal"}],
       "correlations":[{"i":"a","j":"b","r":1.0}],
       "n":20000,"seed":20261003,"idempotency_key":"doc-tiny-r1-0001"}'
# 200

curl -sS http://localhost:8000/api/v1/evaluations/a09a0c76418d425a90c8fccd532465dd   # 200
curl -sS "http://localhost:8000/api/v1/evaluations?model_version=sum-tiny-r1"        # 200
curl -sS http://localhost:8000/api/v1/idempotency/doc-tiny-r1-0001                   # 200
```

小量级不改变非法相关矩阵的处置：给 a、b、c 三个 1e-9 输入指定矛盾的 ±1 组合（最小特征值 −1）仍返回 `422`，且模型不保存（随后 `GET /api/v1/models/...` 为 404）：

```json
{"error": "CorrelationError",
 "message": "相关矩阵不是半正定矩阵：最小特征值 -1.000e+00 < -1e-08",
 "status_code": 422}
```

### 9. 测量精度反求（precision-budgets）

`POST /api/v1/precision-budgets` 是**只读**的 GUM 一阶反求接口：客户端提交**已登记模型版本**、**待调整的单个输入名**和**带单位的非负目标 GUM 合成标准不确定度**；服务读取不可变模型，沿用登记时的单位换算/量纲校验、相关矩阵校验与名义点（中心差分）灵敏度，**仅变动该输入的标准不确定度 u（u ≥ 0）**，其余输入保持登记值，反解

```
u_c²(y) = Σ gᵢgⱼCov(i,j) ≤ 目标²
```

的可行 u 区间。合成方差关于 u 是二次函数（在基本单位空间）：

```
v(u) = C + B·u + A·u²,   A = g_k²,  B = 2·g_k·Σ_{j≠k} g_j·ρ_kj·s_j,  C = Σ_{i≠k,j≠k} g_i g_j Cov(i,j)
```

**所有协方差项（含负相关）都纳入二次不等式。** 零值与舍入误差的裁决全部**按问题自身尺度**进行，绝不用与单位/量级无关的绝对阈值：

* **梯度可辨性（结合差分步长、输出偏置与舍入误差）**：中心差分的初始步长 `h0 = eps^(2/3)·max(|x0|, eps^(2/3))` 极小；当名义点输出带大**偏置**（如 `y = 1 + 1e-4·b`，b = 0）时，差分真值 `g·2h` 可能只有偏置 ulp 的几十倍，被 `eps·max|f(x±h)|` 的百倍门槛判为噪声。此时逐分量按 8 倍几何**自适应放大步长**（上限 `h_cap = eps^(1/3)·max(|x0|, 1)`，即中心差分平衡截断/舍入的最优步长尺度），直到差分值可辨；可辨后继续放大并以相邻步长梯度的稳定性（截断误差 O(h²) 每放大 8 倍增长 64 倍）防止在高曲率点过冲。**扩步被定义域边界截断时**（如 `1.0 + 1e-10*sqrt(x)` 在 x0 = 1e-8 处 x−h < 0，步长到不了 h_cap），改用**域内单侧探针**（只需 x±h 一侧留在域内）放大到 h_cap 确认输出确实依赖该输入——真实斜率（~5e-7）不再被边界处不可辨的中心差分误判为零；斜率值仍取边界内最后一个中心差分估计（单侧一阶差分的 O(h) 截断误差在高曲率点会淹没真实斜率，只用于"是否依赖"的裁决）。只有放大到上限后 `|f+−f-|` 仍不超过输出舍入尺度百倍（如 `2*(a-a)`、`0.0*sqrt(x)` 恒为 0）才判零——真实微小灵敏度（如 `1e-13`，即使远小于其他输入的灵敏度）、被偏置淹没的真实小斜率（如 `1e-4`）与近定义域边界的真实斜率（如 `5e-7`）都**保留**；
* **根吸附**：把理论上为零的根吸到 0 的容差与根自身量级同阶（`eps^(1/3)·根尺度`，无绝对下限）——小量级模型的真实小正根（如 `1e-12` m）不会被吸到 0；
* **端点清理**：只归一 `-0.0` 与防御式负值，真实小正端点（如 `1e-9` mm）原样保留。

响应给出：

* `interval.lower` / `interval.upper`：输入**声明单位**下的端点（`upper` 为 `null` 表示无上限）；
* `interval.lower_inclusive` / `interval.upper_inclusive`：端点开闭性；
* `interval.feasible`：`false` 时为空区间（两端点均为 `null`）；
* `satisfied`：该输入**当前登记**的标准不确定度下，合成标准不确定度是否已达标。

该接口**不修改模型、不写入任何计算记录、不占用幂等键**，可安全重复调用；原有传播、幂等与查询接口不受影响。

#### 9.1 负相关不能误算成单一上限（核心示例）

无量纲线性模型 `y = a + b`（a = 3.0(10)，b = 5.0(20)，r = **−1**）：

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "budget-sum-rm1",
    "name": "线性和 y=a+b（负相关反求示例）",
    "expression": "a + b",
    "inputs": [
      {"name": "a", "value": 3.0, "unit": "dimensionless", "standard_uncertainty": 0.1, "distribution": "normal"},
      {"name": "b", "value": 5.0, "unit": "dimensionless", "standard_uncertainty": 2.0, "distribution": "normal"}
    ],
    "output_unit": "dimensionless",
    "correlations": [{"i": "a", "j": "b", "r": -1.0}]
  }'
```

要求合成标准不确定度不超过 1（反求 a）：

```bash
curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-sum-rm1",
    "input_name": "a",
    "target": {"standard_uncertainty": 1.0, "unit": "dimensionless"}
  }'
```

返回 `200`（以下为**实际响应**）：

```json
{
  "model_version": "budget-sum-rm1",
  "input_name": "a",
  "input_unit": "dimensionless",
  "target": {"standard_uncertainty": 1.0, "unit": "dimensionless"},
  "target_in_output_unit": {"standard_uncertainty": 1.0, "unit": "dimensionless"},
  "sensitivity": {"coefficient": 1.000000000023553, "unit": "dimensionless/dimensionless"},
  "current_standard_uncertainty": {"value": 0.1, "unit": "dimensionless"},
  "current_combined_standard_uncertainty": {"value": 1.900000000015417, "unit": "dimensionless"},
  "satisfied": false,
  "interval": {
    "feasible": true,
    "lower": 0.999999999994219,
    "lower_inclusive": true,
    "upper": 2.999999999947113,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

> 理论上 v(u) = (u − 2)² ≤ 1，即 **ua ∈ [1, 3]**——是一个两侧有界的闭区间，而不是 `[0, 3]` 之类的单一上限（那等价于丢掉负相关的线性项 B·u）。响应端点与 1/3 的 4e-12 级偏差来自中心差分灵敏度（∂y/∂a≈1.000000000024，与 [第 7 节](#7-完全相关r--1奇异半正定矩阵) GUM 偏差同源），不是区间逻辑误差。当前 ua=0.1 时 uc=1.9，故 `satisfied=false`。

#### 9.2 带单位目标 + 相关输入的零端点开区间

以直流功率 `P = U·I`（r(U,I)=0.5，U=10.0(10) V，I=2.00(2) A）为例，要求 u(P) ≤ 0.5 W（反求 U，目标单位直接写 `W`）：

```bash
curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "power-dc-v1",
    "input_name": "U",
    "target": {"standard_uncertainty": 0.5, "unit": "W"}
  }'
```

返回 `200`（**实际响应**）：

```json
{
  "model_version": "power-dc-v1",
  "input_name": "U",
  "input_unit": "V",
  "target": {"standard_uncertainty": 0.5, "unit": "W"},
  "target_in_output_unit": {"standard_uncertainty": 0.5, "unit": "W"},
  "sensitivity": {"coefficient": 1.999999999988437, "unit": "W/V"},
  "current_standard_uncertainty": {"value": 0.1, "unit": "V"},
  "current_combined_standard_uncertainty": {"value": 0.346410161511773, "unit": "W"},
  "satisfied": true,
  "interval": {
    "feasible": true,
    "lower": 0.0,
    "lower_inclusive": false,
    "upper": 0.184520787992712,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

可行区间是 **(0, 0.18452…] V**：

* **低端点 0 标为开**（`lower_inclusive=false`）——U 与 I 在登记模型中有相关系数，而相关系数在零标准不确定度处无定义（登记/传播管线本就拒绝引用 u=0 的相关对，见 [拒绝场景](#6-拒绝场景示例)），因此 u=0 不是可行点；
* 上端点是二次方程根，恒为闭；
* 目标也可用量纲等价单位提交（如 `target.unit="mW"`、`standard_uncertainty=500`），`target_in_output_unit` 给出换算到模型输出单位后的值，端点仍在输入声明单位（V）下。

若目标定得过紧（如 0.2 W）：可行域的数学根退化为只含 u=0，而该零端点对相关输入无效，返回**空区间**：

```json
"interval": {"feasible": false, "lower": null, "lower_inclusive": false,
             "upper": null, "upper_inclusive": false, "unbounded_upper": false}
```

#### 9.3 零灵敏度 → 无上限区间（上界 null）

模型 `y = 2*(a - a) + b` 中 a 对输出的名义点灵敏度恰为 0；a、b 独立，b = 2.0(20)，目标 0.3：

```bash
curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-flat",
    "input_name": "a",
    "target": {"standard_uncertainty": 0.3, "unit": "dimensionless"}
  }'
```

返回 `200`（**实际响应**）：方差与 ua 无关且 0.2 ≤ 0.3，故任意 ua ≥ 0 都可行，区间为 **[0, +∞)**，上界 `null`、`unbounded_upper=true`，当前值 `satisfied=true`：

```json
{
  "sensitivity": {"coefficient": 0.0, "unit": "dimensionless/dimensionless"},
  "current_combined_standard_uncertainty": {"value": 0.19999999999921, "unit": "dimensionless"},
  "satisfied": true,
  "interval": {"feasible": true, "lower": 0.0, "lower_inclusive": true,
               "upper": null, "upper_inclusive": false, "unbounded_upper": true}
}
```

若固定贡献本身已超标（同一模型目标改为 0.1），则无论 ua 取何值都无法达标，返回空区间。

#### 9.4 拒绝场景

| 场景 | 状态码 | 实际响应 |
| --- | --- | --- |
| 未知模型版本 | 404 | `{"error": "NotFoundError", "message": "模型版本 'x' 不存在", "status_code": 404}` |
| 未知输入名 | 422 | `{"error": "RequestError", "message": "输入 'x' 未在模型版本 'budget-sum-rm1' 中登记，可选: ['a', 'b']", "status_code": 422}` |
| 目标单位与输出量纲不符 | 422 | `{"error": "RequestError", "message": "目标单位 'meter' 与模型输出单位 'dimensionless' 量纲不兼容", "status_code": 422}` |
| 目标为负数 | 422 | Pydantic 校验错误数组，`msg` 含「目标标准不确定度必须是非负有限数」 |
| 目标为 NaN / Infinity | 422 | 同上（响应中的非有限输入值已净化为字符串，保证返回体本身是合法 JSON） |
| 请求体缺字段 / 类型错 | 422 | Pydantic 校验错误数组 |

反求接口是幂等只读的：连续重复调用返回完全一致的结果；调用前后 `GET /api/v1/evaluations` 的记录数不变，登记模型的输入标准不确定度也不被修改。反求之后仍可正常提交 `/api/v1/evaluations`（201/200 幂等语义不变）。

#### 9.5 小量级反求：微小灵敏度与小正端点不误判为零

与[第 8 节](#8-小量级标准不确定度1e-9真实方差不得归零)传播管线同理，反求的零值裁决也随问题尺度收缩：微小但真实的灵敏度（`1e-13`）不被相对阈值误判为零，微小但真实的正上界（`1e-9` mm）不被端点清理清零。

**例 1：毫米模型的小正上界。** 直接长度测量 `y = L`（L = 0(2e-9) mm，正态），目标 u(y) ≤ 1e-9 mm：

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "budget-tiny-length",
    "name": "长度直接测量 y=L（小量级反求示例）",
    "expression": "L",
    "inputs": [
      {"name": "L", "value": 0.0, "unit": "millimeter", "standard_uncertainty": 2e-9, "distribution": "normal"}
    ],
    "output_unit": "millimeter"
  }'
# HTTP 201

curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-tiny-length",
    "input_name": "L",
    "target": {"standard_uncertainty": 1e-9, "unit": "millimeter"}
  }'
```

返回 `200`（以下为**实际响应**）：可行区间是闭区间 **[0, 1e-9] mm**——上界 1e-9 mm 是真实正根（基本单位空间 1e-12 m），不会被绝对阈值吸到 0；独立输入的零端点闭。当前 uL = 2e-9 mm 超标，`satisfied=false`：

```json
{
  "model_version": "budget-tiny-length",
  "input_name": "L",
  "input_unit": "mm",
  "target": {"standard_uncertainty": 1e-09, "unit": "millimeter"},
  "target_in_output_unit": {"standard_uncertainty": 1e-09, "unit": "millimeter"},
  "sensitivity": {"coefficient": 1.0, "unit": "millimeter/mm"},
  "current_standard_uncertainty": {"value": 2e-09, "unit": "mm"},
  "current_combined_standard_uncertainty": {"value": 2e-09, "unit": "millimeter"},
  "satisfied": false,
  "interval": {
    "feasible": true,
    "lower": 0.0,
    "lower_inclusive": true,
    "upper": 1e-09,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

**例 2：微小非零灵敏度不是零。** 无量纲模型 `y = a + 1e-13*b`（a = b = 0，ua = 0、ub = 2e9，均正态），目标 u(y) ≤ 1e-4，反求 b。b 的灵敏度 `1e-13` 比 a 的 `1.0` 小 13 个数量级，但它是真实灵敏度：v(u) = (1e-13·u)² ≤ 1e-8 ⇒ **ub ∈ [0, 1e9]**（有限上界），而不是零灵敏度的 `[0, +∞)`：

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "budget-tiny-sens",
    "name": "微小灵敏度 y=a+1e-13*b（小量级反求示例）",
    "expression": "a + 1e-13*b",
    "inputs": [
      {"name": "a", "value": 0.0, "unit": "dimensionless", "standard_uncertainty": 0.0, "distribution": "normal"},
      {"name": "b", "value": 0.0, "unit": "dimensionless", "standard_uncertainty": 2e9, "distribution": "normal"}
    ],
    "output_unit": "dimensionless"
  }'
# HTTP 201

curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-tiny-sens",
    "input_name": "b",
    "target": {"standard_uncertainty": 1e-4, "unit": "dimensionless"}
  }'
```

返回 `200`（以下为**实际响应**；上界 `1000000000.0000001` 即 1e9，尾差来自中心差分灵敏度的 ~1e-16 相对舍入）：

```json
{
  "model_version": "budget-tiny-sens",
  "input_name": "b",
  "input_unit": "dimensionless",
  "target": {"standard_uncertainty": 0.0001, "unit": "dimensionless"},
  "target_in_output_unit": {"standard_uncertainty": 0.0001, "unit": "dimensionless"},
  "sensitivity": {"coefficient": 1e-13, "unit": "dimensionless/dimensionless"},
  "current_standard_uncertainty": {"value": 2000000000.0, "unit": "dimensionless"},
  "current_combined_standard_uncertainty": {"value": 0.0002, "unit": "dimensionless"},
  "satisfied": false,
  "interval": {
    "feasible": true,
    "lower": 0.0,
    "lower_inclusive": true,
    "upper": 1000000000.0000001,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

> 当前 ub = 2e9 时 uc = 2e-4 > 1e-4，故 `satisfied=false`；把 ub 收紧到 1e9 以内即达标。若 b 的灵敏度是真零（如 `2*(a-a)+b` 中的 a），同样的请求仍返回 `[0, +∞)`（上界 `null`、`unbounded_upper=true`，见 [9.3](#93-零灵敏度--无上限区间上界-null)）——零与非零按梯度自身舍入尺度裁决，与灵敏度的绝对大小无关。

#### 9.6 输出偏置淹没小灵敏度：自适应步长恢复梯度（闭区间 ~5000）

无量纲模型 `y = a + 1e-4*b`（a = 1(0)，b = 0(10000)，独立，正态）。输出在名义点带偏置 y = 1，而 b 的真实灵敏度只有 `1e-4`；要求 u(y) ≤ 0.5，反求 b。

在零名义点 b = 0 上，中心差分的初始步长 `h0 = eps^(2/3) ≈ 3.67e-11` 极小，差分真值 `1e-4·2h0 ≈ 7.34e-15` 只有偏置 `1.0` 一个 ulp（≈2.22e-16）的 **33 倍**，低于百倍噪声门槛——固定步长实现会把这个真实非零梯度误判为零，尽管响应里的灵敏度系数并非 0，却把有限预算报成无上界 `[0, +∞)`。服务结合**差分步长、输出偏置与舍入误差**把步长自适应放大到 `h_cap = eps^(1/3) ≈ 6.06e-6`（中心差分最优步长尺度），该步长上差分值约为偏置 ulp 的 5.5 万倍，梯度清晰可辨，二次型 `v(u) = (1e-4·u)² ≤ 0.25` 正常求解。

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "budget-bias",
    "name": "带偏置的小灵敏度 y=a+1e-4*b",
    "expression": "a + 1e-4*b",
    "inputs": [
      {"name": "a", "value": 1.0, "unit": "dimensionless", "standard_uncertainty": 0.0, "distribution": "normal"},
      {"name": "b", "value": 0.0, "unit": "dimensionless", "standard_uncertainty": 10000.0, "distribution": "normal"}
    ],
    "output_unit": "dimensionless"
  }'
# HTTP 201

curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-bias",
    "input_name": "b",
    "target": {"standard_uncertainty": 0.5, "unit": "dimensionless"}
  }'
```

返回 `200`（以下为**实际响应**）：可行区间是闭区间 **[0, 5000.0004…]**（理论 ub ≤ 5000，尾差来自中心差分灵敏度 9.9999991e-5 与 `1e-4` 的 ~9e-7 相对舍入），当前 ub = 10000 时 uc ≈ 1.0 超标，`satisfied=false`：

```json
{
  "model_version": "budget-bias",
  "input_name": "b",
  "input_unit": "dimensionless",
  "target": {"standard_uncertainty": 0.5, "unit": "dimensionless"},
  "target_in_output_unit": {"standard_uncertainty": 0.5, "unit": "dimensionless"},
  "sensitivity": {
    "coefficient": 9.9999991143e-05,
    "unit": "dimensionless/dimensionless"
  },
  "current_standard_uncertainty": {"value": 10000.0, "unit": "dimensionless"},
  "current_combined_standard_uncertainty": {"value": 0.999999911432388, "unit": "dimensionless"},
  "satisfied": false,
  "interval": {
    "feasible": true,
    "lower": 0.0,
    "lower_inclusive": true,
    "upper": 5000.000442838098,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

> 若斜率是**真零**（如偏置模型 `y = 1 + x*x` 在 x = 0 处，放大到 `h_cap` 后差分值仍为偏置舍入噪声），同样的请求依旧返回 `[0, +∞)`；零基值上的真实微小梯度（[9.5 例 2](#95-小量级反求微小灵敏度与小正端点不误判为零)）、单位换算、负相关双端区间（[9.1](#91-负相关不能误算成单一上限核心示例)）等原有行为均不受影响。

#### 9.7 定义域边界截断差分扩步：单侧探针确认斜率（闭区间 ~0.2）

无量纲模型 `y = 1.0 + 1e-10*sqrt(x)`（x = 1e-8(1e-9)，正态），目标 u(y) ≤ 1e-7，反求 x。真实斜率 `1e-10/(2·sqrt(1e-8)) = 5e-7`，理论可行区间 **ux ∈ [0, 0.2]**。

这里中心差分的自适应扩步被**定义域边界**截断：步长 h 稍大于 x0 = 1e-8 时 x−h < 0，`sqrt` 产生 NaN，扩步停止在 h ≈ 2.3e-9（到不了最优步长 `h_cap ≈ 6.1e-6`）；该步长上的差分值 `g·2h ≈ 2.4e-15` 达不到偏置 1.0 的百倍舍入门槛（≈2.2e-14）——只按中心差分噪声门槛裁决会把真实斜率误判为零，错误返回无上界 `upper=null`。

服务此时改用**域内单侧探针**（只需 x+h 一侧留在域内）从当前步长继续放大到 `h_cap`：h ≈ 1.5e-7 时 `|f(x+h)−f(x)| ≈ 3e-14` 越过同一可辨门槛，确认输出确实依赖 x；**斜率值仍取边界内最后一个中心差分估计**（≈5.2e-7）——单侧一阶差分的 O(h) 截断误差在 sqrt 近 0 的高曲率处会淹没甚至反转真实斜率，只能用于"是否依赖"的裁决。随后二次型 `v(u) = (g·u)² ≤ 1e-14` 正常求解。

```bash
curl -sS -X POST http://localhost:8000/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "budget-sqrt-boundary",
    "name": "近定义域边界的小斜率 y=1+1e-10*sqrt(x)",
    "expression": "1.0 + 1e-10*sqrt(x)",
    "inputs": [
      {"name": "x", "value": 1e-8, "unit": "dimensionless", "standard_uncertainty": 1e-9, "distribution": "normal"}
    ],
    "output_unit": "dimensionless"
  }'
# HTTP 201

curl -sS -X POST http://localhost:8000/api/v1/precision-budgets \
  -H 'Content-Type: application/json' \
  -d '{
    "model_version": "budget-sqrt-boundary",
    "input_name": "x",
    "target": {"standard_uncertainty": 1e-7, "unit": "dimensionless"}
  }'
```

返回 `200`（以下为**实际响应**）：可行区间是闭区间 **[0, 0.19216…]**（理论 0.2；偏差来自边界内中心差分估计 5.2039e-7 与精确斜率 5e-7 的 ~4% 舍入），当前 ux = 1e-9 时 uc ≈ 5.2e-16 ≤ 1e-7，`satisfied=true`：

```json
{
  "model_version": "budget-sqrt-boundary",
  "input_name": "x",
  "input_unit": "dimensionless",
  "target": {"standard_uncertainty": 1e-07, "unit": "dimensionless"},
  "target_in_output_unit": {"standard_uncertainty": 1e-07, "unit": "dimensionless"},
  "sensitivity": {"coefficient": 5.20390617e-07, "unit": "dimensionless/dimensionless"},
  "current_standard_uncertainty": {"value": 1e-09, "unit": "dimensionless"},
  "current_combined_standard_uncertainty": {"value": 5.20390617002552e-16, "unit": "dimensionless"},
  "satisfied": true,
  "interval": {
    "feasible": true,
    "lower": 0.0,
    "lower_inclusive": true,
    "upper": 0.19216334179121,
    "upper_inclusive": true,
    "unbounded_upper": false
  }
}
```

> 同一边界结构下的**真零斜率**（如 `y = 1.0 + 0.0*sqrt(x)`）单侧探针到 `h_cap` 差值恒为 0，仍判零：同样的请求返回 `[0, +∞)`（`upper=null`、`unbounded_upper=true`）——边界截断不改变真零方向的裁决。反求与 GUM 传播共用同一梯度路径：该模型的 `POST /api/v1/evaluations` 给出的 GUM 合成标准不确定度与上面响应中的 `current_combined_standard_uncertainty` 完全一致（5.20390617002552e-16）。

## 模型表达式白名单

表达式在服务端经 Python AST 解析后**逐节点白名单校验**，不经过任何 `eval`：

- **数字**：仅十进制 int/float（禁止 bool、complex、字符串）；
- **变量**：仅登记模型中声明的输入名（标识符格式，不能叫 `pi`/`e`）；
- **常量**：`pi`、`e`；
- **算子**：`+ - * / **` 及一元 `+ -`；
- **一元函数**：`sqrt abs exp log log10 sin cos tan asin acos atan sinh cosh tanh`；
- **二元函数**：`atan2(y, x)`（两参数量纲必须相同，返回无量纲）；
- 三角函数/指数/对数的参数必须是**无量纲量**（带角度参与运算请显式处理，如 `sin(theta_rad)`）。

登记时会立即执行：语法校验 → 变量齐备性检查 → Pint 量纲推导 → 输出单位量纲核对 → 相关矩阵半正定校验。

模型**不可变**：同一 `version` 再次登记返回 409；需要修改表达式、输入集合或不确定度预算时请登记新版本。

## 分布与相关系数

每个输入登记一个分布，标准不确定度 `u(x)` 对应该分布的总体标准差：

| 分布 | 含义 | 内部抽样参数 |
| --- | --- | --- |
| `normal` | 正态 N(x, u²) | x + u·z |
| `uniform` | 以 x 为中心、宽度 2a 的均匀分布，u = a/√3 | x + u·√3·(2v−1) |
| `triangular` | 对称三角分布（支撑 ±a，峰值在中心），u = a/√6 | x + u·√6·T，T∈[−1,1] |

相关性通过 Gaussian copula 引入：先用相关矩阵因子（正定走 Cholesky；奇异半正定走特征分解对称平方根）生成相关标准正态，再经正态 CDF 与目标分布分位数函数映射，因此**边际分布精确且相关结构正确**。r = ±1 的完全相关（含零特征值的奇异相关矩阵）同样支持：正态边际的两个标准化输入固定种子下严格同向/反向变化。

## 结果字段说明

- `combined.gum`：GUM 一阶结果——`value` 合成值、`standard_uncertainty` 合成标准不确定度、`coverage_factor=1.95996` 与 95% 覆盖区间；
- `combined.monte_carlo`：MC 结果——`mean`、`median`、`standard_uncertainty`、2.5/97.5 分位 `coverage_interval`、抽样极值；
- `sensitivities`：中心差分解析得到的灵敏度系数 ∂y/∂xᵢ，附单位（输出单位/输入单位）；
- `contributions.gum`：逐输入方差贡献（`individual`）与成对协方差贡献（`covariance`），含占总方差比例 `fraction`（相关时比例之和可不为 1）；
- `contributions.monte_carlo`：各输入抽样与输出抽样的 Pearson r 与 r²（蒙特卡洛方差重要性诊断）；
- `convergence`：收敛统计——
  - `se_mean = s/√N`（均值标准误）；
  - `se_std = s·√((γ₂+2)/(4N))`（标准差标准误，γ₂ 为超额峰度）；
  - `se_quantile_*`：用 Silverman 高斯核密度估计的 2.5%/97.5% 分位标准误 `√(p(1−p)/N)/f(q)`；
  - `rel_se_*`：相对尺度（均值以 max(|mean|, s) 为分母，分位以区间半宽为分母）；
  - 三个相对标准误均不超过 `convergence_tolerance`（默认 0.05，范围 (0, 0.5]）才判为收敛。不收敛返回 422，可用**更大的 N + 新幂等键**重试。

## 幂等与“不保存部分结果”语义

- 每个计算请求在服务端规范化为 SHA-256 `request_hash`（规范化输入快照按名排序、数值统一在声明单位下表示），`idempotency_key` 上也有唯一约束；
- 计算在数据库事务之外完成：只有 GUM 与 MC 全部成功且收敛后才执行**唯一一次**插入（插入事务内再复查唯一键）。因此任何 422/4xx 都不会留下部分结果或占用幂等键；
- 同一幂等键重复提交相同请求 → 200 + `cached=true`；相同键不同载荷 → 409。

## 运行测试

```bash
pip install -r requirements.txt pytest httpx
python -m pytest -q
```

测试覆盖：表达式注入拒绝、单位换算与量纲校验、PSD 拒绝（< -1e-8）/舍入修复（[-1e-8,0)，保单位对角）/奇异接受（r=±1 零特征值）、r=±1 的 GUM 解析解 |ua+r·ub| 与 MC 同/反向传播、完全抵消退化为确定值、**小量级（u=1e-9）真实 ~1e-18 方差不被固定绝对阈值钳零（r=+1→2e-9、r=−1 不等→1e-9、r=−1 相等→0）及其 HTTP 登记/计算/幂等/查询/非法矩阵拒绝不落库**、MC 与线性高斯解析解对比、固定种子可复现、幂等与冲突、不收敛/NaN 不落库、查询接口、**精度反求二次不等式（负相关 r=−1→ua∈[1,3] 双侧有界、相关输入零端点开、零灵敏度→上界 null 无上限、无法达标→空区间、目标零/单位换算/未知版本/未知输入/量纲不符/负值/NaN/Inf 拒绝、只读不改模型不写记录不影响幂等与传播）**、**小量级反求（毫米模型目标 1e-9→闭区间 [0,1e-9] mm 真实小正上界保留、1e-13 微小灵敏度不误判为零→[0,1e9] 有限上界、小量级负相关双端区间 [1e-9,3e-9]、小量级真零灵敏度仍无上限）**、**定义域边界反求（1+1e-10·sqrt(x) 在 x0=1e-8 处中心差分扩步被 x−h<0 截断→域内单侧探针确认真实斜率 5e-7→闭区间 [0,~0.19] 且当前值达标、同边界真零斜率 0.0*sqrt(x) 仍无上限、反求与 GUM 管线梯度一致、过紧目标→上界随目标收缩且不达标）**等（88 个用例）。

## 项目结构

```
.
├── app/
│   ├── main.py          # FastAPI 路由、错误映射、生命周期
│   ├── models.py        # Pydantic 请求模型与参数约束（含精度反求请求）
│   ├── expressions.py   # AST 白名单校验 + NumPy 求值
│   ├── units.py         # Pint 单位换算、量纲推导、基本单位缩放
│   ├── engine.py        # 相关矩阵、分布抽样、GUM、蒙特卡洛与收敛判据
│   ├── services.py      # 模型登记、快照核对、计算管线编排、精度反求二次不等式
│   ├── storage.py       # SQLite（WAL、写锁、幂等插入）
│   ├── errors.py        # 领域错误（统一 4xx 映射）
│   └── config.py        # 环境变量配置
├── tests/               # pytest 端到端 + 数值正确性测试
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py
```
