"""Standalone, resource-limited evaluator for local math-program validation.

Run with Python -I -S -B, never inside the training process. The restricted
builtins/imports are a best-effort guard, not a general-purpose OS sandbox.
"""

import ast
import builtins
import contextlib
import decimal
import fractions
import json
import math
import resource
import statistics
import sys
import types


def evaluate(code, timeout_seconds, memory_mb):
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout_seconds) + 1,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)

    modules = {}
    exports = {
        "math": {
            name: getattr(math, name) for name in dir(math) if not name.startswith("_")
        },
        "fractions": {"Fraction": fractions.Fraction},
        "decimal": {
            name: getattr(decimal, name)
            for name in ("Decimal", "getcontext", "localcontext")
        },
        "statistics": {name: getattr(statistics, name) for name in statistics.__all__},
    }
    for name, values in exports.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            level
            or name not in modules
            or any(part.startswith("_") for part in (fromlist or ()))
        ):
            raise ImportError("Only public math-library imports are supported.")
        return modules[name]

    try:
        if len(code) > 32768:
            return None
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                return None
            if (
                isinstance(node, ast.Name)
                and node.id.startswith("__")
                and node.id != "__name__"
            ):
                return None
            if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
                return None
        names = (
            "abs all any bool dict divmod enumerate filter float int isinstance "
            "len list map max min next pow print range repr reversed round set "
            "sorted str sum tuple zip Exception ValueError TypeError "
            "ZeroDivisionError OverflowError StopIteration"
        ).split()
        allowed = {name: getattr(builtins, name) for name in names}
        allowed["__import__"] = restricted_import
        namespace = {"__builtins__": allowed, "__name__": "__validation__"}

        class DiscardOutput:
            def write(self, text):
                return len(text)

            def flush(self):
                pass

        sink = DiscardOutput()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(tree, "<validation-program>", "exec"), namespace)
            result = namespace["solve"]()
            if isinstance(result, bool) or not isinstance(
                result, (int, float, decimal.Decimal, fractions.Fraction)
            ):
                return None
            result = float(result)
        return result if math.isfinite(result) else None
    except BaseException:
        return None


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    result = evaluate(**request)
    print(json.dumps(result, allow_nan=False))
