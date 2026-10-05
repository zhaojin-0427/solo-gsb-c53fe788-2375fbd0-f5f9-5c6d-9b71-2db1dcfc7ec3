"""安全模型表达式：AST 白名单校验 + NumPy 求值 + Pint 维度推导。

表达式只允许：
  * 数字（int/float，禁止 bool/complex/字符串）
  * 模型登记时声明的输入变量名
  * 常量 ``pi``、``e``
  * 二元算子  + - * / **
  * 一元算子  + -
  * 白名单函数（见 :data:`FUNCTIONS`）

任何属性访问、下标、比较、布尔运算、名字（lambda/推导式/内置名）都会被拒绝，
从而从根本上杜绝 ``__import__`` / ``open`` 等注入。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import ExpressionError

# ---------------------------------------------------------------------------
# 白名单
# ---------------------------------------------------------------------------

#: 允许的一元函数：名称 -> NumPy 实现（均支持向量化）
FUNCTIONS: dict[str, Any] = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}

#: 允许的常量（无量纲）
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


@dataclass(frozen=True)
class CompiledExpression:
    """校验后的表达式（不可变）。"""

    source: str
    tree: ast.Expression
    variables: tuple[str, ...]

    def evaluate(self, env: Mapping[str, Any], funcs: Mapping[str, Any] | None = None) -> Any:
        """在给定环境下求值。

        :param env: 变量/常量值。
        :param funcs: 覆盖函数表（Pint 维度推导时传入包过单位的版本）。
        """
        fns = FUNCTIONS if funcs is None else funcs
        return _eval_node(self.tree.body, env, fns)


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------

def compile_expression(source: str, variables: tuple[str, ...]) -> CompiledExpression:
    """解析并按白名单校验表达式。"""
    if not isinstance(source, str) or not source.strip():
        raise ExpressionError("模型表达式不能为空")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"表达式语法错误: {exc.msg}") from exc

    allowed = set(variables) | set(CONSTANTS)
    used: set[str] = set()
    _validate(tree, allowed, used)

    missing = set(variables) - used
    if missing:
        raise ExpressionError(f"表达式未使用已登记的输入量: {sorted(missing)}")

    return CompiledExpression(source=source, tree=tree, variables=tuple(sorted(used & set(variables))))


def _validate(tree: ast.Expression, allowed_names: set[str], used: set[str]) -> None:
    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("表达式只允许整型/浮点型数字常量")
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ExpressionError(f"表达式含未声明的名字: {node.id!r}")
            used.add(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ExpressionError(f"不允许的二元算子: {type(node.op).__name__}")
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ExpressionError(f"不允许的一元算子: {type(node.op).__name__}")
            visit(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("只允许调用白名单中的简单函数名")
            if node.func.id not in FUNCTIONS:
                raise ExpressionError(f"函数不在白名单中: {node.func.id!r}")
            if node.keywords:
                raise ExpressionError("函数调用不允许关键字参数")
            arity = 2 if node.func.id == "atan2" else 1
            if len(node.args) != arity:
                raise ExpressionError(f"函数 {node.func.id} 需要 {arity} 个位置参数")
            for arg in node.args:
                visit(arg)
        else:
            raise ExpressionError(f"表达式含不允许的结构: {type(node).__name__}")

    visit(tree)


# ---------------------------------------------------------------------------
# 求值
# ---------------------------------------------------------------------------

def _eval_node(node: ast.AST, env: Mapping[str, Any], funcs: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env, funcs)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        a = _eval_node(node.left, env, funcs)
        b = _eval_node(node.right, env, funcs)
        return _BIN_OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, env, funcs))
    if isinstance(node, ast.Call):
        args = [_eval_node(a, env, funcs) for a in node.args]
        return funcs[node.func.id](*args)
    # 校验阶段已经排除，理论不可达
    raise ExpressionError(f"不允许的 AST 节点: {type(node).__name__}")


def numpy_evaluation(compiled: CompiledExpression, values: Mapping[str, np.ndarray | float]) -> np.ndarray:
    """NumPy 标量/数组求值（蒙特卡洛与名义值计算共用）。"""
    env: dict[str, Any] = dict(CONSTANTS)
    env.update(values)
    try:
        result = compiled.evaluate(env)
    except Exception as exc:  # numpy 对负数开方等会产生 warning + nan，这里只捕获真异常
        from .errors import PropagationError

        raise PropagationError(f"模型求值失败: {exc}") from exc
    arr = np.asarray(result, dtype=float)
    return arr
