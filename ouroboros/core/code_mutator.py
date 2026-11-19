"""Source-edit proposals and bounded numeric-objective experiments.

Syntax validation is NOT a security sandbox. Generated source is exported as a reviewable
patch and never imported, executed, committed, or merged automatically.
"""

import ast
import difflib
import hashlib
import math
from pathlib import Path

from .checkpoints import atomic_json


class CodeMutator:
    def __init__(self, target_filepath=None):
        self.target_filepath = target_filepath
        self.mutation_log = []

    def propose_loss_mutation(self, current_code, adjustment_type, new_value):
        if not math.isfinite(new_value) or new_value < 0:
            raise ValueError("Mutation values must be finite and nonnegative")
        if adjustment_type in {"dpo_beta", "learning_rate"} and new_value == 0:
            raise ValueError("Learning rate and DPO beta must be positive")
        if adjustment_type == "lora_dropout" and new_value >= 1:
            raise ValueError("Dropout must be less than 1")
        names = {
            "dpo_beta": "beta",
            "learning_rate": "learning_rate",
            "lora_dropout": "lora_dropout",
        }
        if adjustment_type not in names:
            raise ValueError(f"Unsupported mutation: {adjustment_type}")
        tree = ast.parse(current_code)
        target, matches = names[adjustment_type], []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == target:
                matches.append(node.value)
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == target
            ):
                matches.append(node.value)
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target for t in node.targets
            ):
                matches.append(node.value)
        if (
            len(matches) != 1
            or not isinstance(matches[0], ast.Constant)
            or type(matches[0].value) not in (int, float)
        ):
            raise ValueError("Expected exactly one numeric literal at the mutation target")
        value = matches[0]
        # AST columns are UTF-8 byte offsets, not Python character indices.
        lines = current_code.splitlines(keepends=True)
        prefix = sum(len(line.encode()) for line in lines[: value.lineno - 1]) + value.col_offset
        suffix = (
            sum(len(line.encode()) for line in lines[: value.end_lineno - 1]) + value.end_col_offset
        )
        replacement = (
            f"{new_value:.3f}" if adjustment_type == "dpo_beta" else repr(float(new_value))
        )
        raw = current_code.encode()
        mutated = (raw[:prefix] + replacement.encode() + raw[suffix:]).decode()
        if not self.validate_syntax(mutated):
            raise ValueError("Mutation produced invalid Python")
        self.mutation_log.append(
            {"type": adjustment_type, "new_value": new_value, "status": "proposed"}
        )
        return mutated

    @staticmethod
    def validate_syntax(code_string):
        try:
            ast.parse(code_string)
            return True
        except (SyntaxError, ValueError):
            return False

    def export_proposal(
        self, current_code, proposed_code, output_dir, *, relative_path="objective.py"
    ):
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Proposal path must be a relative .py source file")
        if not self.validate_syntax(proposed_code):
            raise ValueError("Proposed code is not valid Python")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        patch = "".join(
            difflib.unified_diff(
                current_code.splitlines(keepends=True),
                proposed_code.splitlines(keepends=True),
                fromfile=f"a/{path.as_posix()}",
                tofile=f"b/{path.as_posix()}",
            )
        )
        (output / "proposal.patch").write_text(patch, encoding="utf-8")
        atomic_json(
            output / "proposal.json",
            {
                "target": path.as_posix(),
                "base_sha256": hashlib.sha256(current_code.encode()).hexdigest(),
                "proposed_sha256": hashlib.sha256(proposed_code.encode()).hexdigest(),
                "syntax_valid": True,
                "executed": False,
                "approved": False,
                "warning": "Syntax is not isolation. Test untrusted source in an OS/container sandbox before review.",
            },
        )
        return str(output / "proposal.patch")


class NumericObjective:
    """A deliberately small arithmetic language, interpreted WITHOUT eval/exec/compile.

    Enables loss scaling or schedules such as loss * beta / (1 + step), not arbitrary Python.
    Constants/nodes/exponents and all intermediate results are bounded.
    """

    def __init__(self, expression, variables=("loss", "beta", "step")):
        if len(expression) > 1000:
            raise ValueError("Objective is too long")
        try:
            self.tree = ast.parse(expression, mode="eval").body
        except SyntaxError as exc:
            raise ValueError("Invalid objective syntax") from exc
        self.variables = set(variables)
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        )
        nodes = list(ast.walk(self.tree))
        if len(nodes) > 64 or any(not isinstance(node, allowed) for node in nodes):
            raise ValueError("Only bounded scalar arithmetic is permitted")
        if any(isinstance(node, ast.Name) and node.id not in self.variables for node in nodes):
            raise ValueError("Unknown objective variable")
        for node in nodes:
            if isinstance(node, ast.Constant) and (
                type(node.value) not in (int, float)
                or abs(node.value) > 1e6
                or not math.isfinite(node.value)
            ):
                raise ValueError("Constants must be finite numbers with magnitude <= 1e6")
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) not in (int, float)
                    or not 0 <= node.right.value <= 4
                )
            ):
                raise ValueError("Exponents must be literal values between 0 and 4")

    def __call__(self, **values):
        if set(values) != self.variables or any(
            type(v) not in (int, float) or abs(v) > 1e12 or not math.isfinite(v)
            for v in values.values()
        ):
            raise ValueError("Supply exactly the declared, finite bounded scalar variables")

        def visit(node):
            if isinstance(node, ast.Constant):
                result = float(node.value)
            elif isinstance(node, ast.Name):
                result = float(values[node.id])
            elif isinstance(node, ast.UnaryOp):
                result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            else:
                a, b = visit(node.left), visit(node.right)
                operations = {
                    ast.Add: lambda: a + b,
                    ast.Sub: lambda: a - b,
                    ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / b,
                    ast.Pow: lambda: a**b,
                }
                try:
                    result = operations[type(node.op)]()
                except (ZeroDivisionError, OverflowError) as exc:
                    raise ValueError("Invalid objective arithmetic") from exc
            if isinstance(result, complex) or not math.isfinite(result) or abs(result) > 1e12:
                raise ValueError("Objective result exceeds numeric bounds")
            return result

        return visit(self.tree)
