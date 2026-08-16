"""Safe, AST-validated expression evaluation for the GLPlot GUI.

This module turns user-typed text such as ``a * sin(b * x) + noise(x, 0.1)`` into a numpy
array. It is the only place in the GUI where user text becomes executable code, and it is
therefore written as a security boundary rather than as a convenience wrapper around
:func:`eval`.

Design
------
Everything is rejected at **parse time** by an allowlist. Nothing relies on catching a bad
call at runtime.

* Only a fixed set of AST node types survives :func:`validate`; every other node type is a
  hard error. In particular :class:`ast.Attribute` is banned outright, so there is no dotted
  access of any kind. That single rule kills the whole ``().__class__.__bases__[0].
  __subclasses__()`` family of sandbox escapes at the parser, not at the interpreter.
* Any identifier containing ``__`` is rejected (defense in depth behind the Attribute ban).
* A :class:`ast.Call` may only target a *bare name* that is present in :data:`SAFE_NAMES`
  and callable. Calling a variable, a subscript or a returned value is impossible.
* ``eval`` runs with ``{"__builtins__": {}}`` as globals, so a name that is neither in
  :data:`SAFE_NAMES` nor in the caller's ``variables`` cannot reach the builtins at all --
  it raises :class:`NameError`, which is wrapped into :class:`ExpressionError`.
* CPU/memory bombs are bounded: expressions longer than ``MAX_EXPR_CHARS``, string constants
  longer than ``MAX_STRING_CHARS``, and ``**`` with a statically-known exponent above
  ``MAX_POW_EXPONENT`` are rejected. The exponent is constant-folded in *float* arithmetic,
  so ``9 ** 9 ** 9`` is rejected without the validator itself ever materialising a bignum.

Deviations from the interface spec (deliberate, documented)
----------------------------------------------------------
1. ``validate()`` accepts an optional keyword-only ``names`` allowlist. When it is ``None``
   (the default) an unknown *bare* name is allowed through as a free variable, because that
   is what makes ``sin(x)`` and the parameter sliders of the functions panel work at all --
   ``x``, ``a``, ``b`` are not in :data:`SAFE_NAMES`. This is not a security relaxation: an
   unbound name can only ever resolve to :data:`SAFE_NAMES` or the caller-supplied
   ``variables``, and with empty builtins it otherwise raises :class:`NameError`. Pass
   ``names=(...)`` to get the strict behaviour where every name must be known.
2. :class:`ast.keyword` is allowed, otherwise ``gauss(x, mu=0, sigma=1)`` -- which the spec
   requires -- could not be typed. ``**kwargs`` unpacking (a ``keyword`` with ``arg=None``)
   is still rejected, as is ``*args`` unpacking (via the :class:`ast.Starred` ban).

Note on ``min``/``max`` versus ``maximum``/``minimum``: ``min``/``max`` are numpy
*reductions* (``min(y)`` -> one number), while ``maximum``/``minimum`` are *elementwise*
(``maximum(y, 0)`` -> an array). The functions panel help text must say so.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

import numpy as np

__all__ = [
    "ExpressionError",
    "SAFE_NAMES",
    "compile_1d",
    "evaluate",
    "evaluate_1d",
    "free_variables",
    "validate",
]


class ExpressionError(Exception):
    """Raised for any expression that is rejected, or that fails while evaluating."""


# --- limits -------------------------------------------------------------------------------

#: Longest accepted expression, in characters.
MAX_EXPR_CHARS = 8192
#: Longest accepted string/bytes constant, in characters.
MAX_STRING_CHARS = 4096
#: Largest accepted statically-known ``**`` exponent.
MAX_POW_EXPONENT = 1.0e6
#: Largest number of samples ``noise(n)`` will allocate.
MAX_NOISE_SAMPLES = 100_000_000


# --- convenience helpers exposed to users -------------------------------------------------


def gauss(x, mu=0.0, sigma=1.0):
    """Unit-amplitude Gaussian bell ``exp(-0.5 * ((x - mu) / sigma) ** 2)``.

    Peaks at exactly 1.0 -- this is the curve you want to *plot*, not the normalised
    probability density. Raises ValueError when ``sigma`` is zero.
    """
    xa = np.asarray(x, dtype=np.float64)
    mua = np.asarray(mu, dtype=np.float64)
    siga = np.asarray(sigma, dtype=np.float64)
    if np.any(siga == 0.0):
        raise ValueError("gauss(): sigma must be non-zero")
    return np.exp(-0.5 * ((xa - mua) / siga) ** 2)


def sigmoid(x):
    """Logistic sigmoid ``1 / (1 + exp(-x))``.

    Computed as ``0.5 * (1 + tanh(x / 2))``, which is mathematically identical but does not
    overflow for large negative ``x``.
    """
    xa = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * xa))


def rect(x):
    """Rectangular pulse: 1.0 for ``|x| < 0.5``, 0.5 at ``|x| == 0.5``, else 0.0."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(ax < 0.5, 1.0, np.where(ax == 0.5, 0.5, 0.0))


def step(x):
    """Unit step: 0.0 for ``x < 0``, 1.0 for ``x >= 0``."""
    return np.heaviside(np.asarray(x, dtype=np.float64), 1.0)


def heaviside(x, x2=0.5):
    """Heaviside step with a defaulted value at the discontinuity.

    Identical to ``np.heaviside`` except that ``x2`` (the value at ``x == 0``) defaults to
    0.5 instead of being required, so ``heaviside(x)`` is typeable.
    """
    return np.heaviside(np.asarray(x, dtype=np.float64), x2)


def sawtooth(x, period=1.0):
    """Rising sawtooth in ``[-1, 1)``, zero-crossing at ``x == 0``, repeating every ``period``."""
    xa = np.asarray(x, dtype=np.float64)
    pa = np.asarray(period, dtype=np.float64)
    if np.any(pa == 0.0):
        raise ValueError("sawtooth(): period must be non-zero")
    t = xa / pa
    return 2.0 * (t - np.floor(t + 0.5))


def square(x, period=1.0):
    """Square wave in ``{-1, 1}``: +1 over the first half of each ``period``, -1 over the second."""
    xa = np.asarray(x, dtype=np.float64)
    pa = np.asarray(period, dtype=np.float64)
    if np.any(pa == 0.0):
        raise ValueError("square(): period must be non-zero")
    return np.where(np.mod(xa / pa, 1.0) < 0.5, 1.0, -1.0)


def noise(n_or_x, scale=1.0, seed=None):
    """Gaussian noise with standard deviation ``scale``.

    ``n_or_x`` is either a sample count (``noise(500)``) or an array whose shape is matched
    (``noise(x, 0.1)``). ``seed`` makes the draw reproducible.
    """
    arr = np.asarray(n_or_x)
    if arr.ndim == 0:
        count = float(arr)
        if not np.isfinite(count) or count < 0.0:
            raise ValueError("noise(): sample count must be a non-negative finite number")
        if count > MAX_NOISE_SAMPLES:
            raise ValueError(
                f"noise(): sample count {count:.0f} exceeds the limit of {MAX_NOISE_SAMPLES}"
            )
        shape: Any = (int(count),)
    else:
        if arr.size > MAX_NOISE_SAMPLES:
            raise ValueError(
                f"noise(): {arr.size} samples exceeds the limit of {MAX_NOISE_SAMPLES}"
            )
        shape = arr.shape
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, float(scale), size=shape)


# --- the user-visible namespace -----------------------------------------------------------

#: Every name a user expression may reference. Bare callables and scalars only -- no module
#: objects. Exposing ``np`` itself would be pointless (Attribute access is banned) and risky.
SAFE_NAMES: Dict[str, Any] = {
    # trigonometric / hyperbolic
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    # exponential / logarithmic
    "exp": np.exp,
    "expm1": np.expm1,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "power": np.power,
    # rounding / sign
    "abs": np.abs,
    "fabs": np.fabs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "trunc": np.trunc,
    "clip": np.clip,
    "mod": np.mod,
    # statistics / reductions
    "sum": np.sum,
    "cumsum": np.cumsum,
    "prod": np.prod,
    "cumprod": np.cumprod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "argmin": np.argmin,
    "argmax": np.argmax,
    "ptp": np.ptp,
    # construction
    "linspace": np.linspace,
    "arange": np.arange,
    "zeros": np.zeros,
    "ones": np.ones,
    "full": np.full,
    "array": np.array,
    # selection (elementwise)
    "where": np.where,
    "maximum": np.maximum,
    "minimum": np.minimum,
    # constants
    "pi": float(np.pi),
    "e": float(np.e),
    "tau": float(math.tau),
    "inf": float(np.inf),
    "nan": float(np.nan),
    # signal-ish
    "heaviside": heaviside,
    "sinc": np.sinc,
    "gradient": np.gradient,
    "diff": np.diff,
    "interp": np.interp,
    # convenience
    "gauss": gauss,
    "sigmoid": sigmoid,
    "rect": rect,
    "step": step,
    "sawtooth": sawtooth,
    "square": square,
    "noise": noise,
}


# --- the AST allowlist --------------------------------------------------------------------


def _node_types(*names: str) -> Set[type]:
    """Resolve AST class names to types, skipping those absent on this Python version.

    ``ast.Index`` is deprecated (3.9) and slated for removal; resolving by name keeps the
    allowlist working across the 3.9-3.12 support matrix without a version check.
    """
    found: Set[type] = set()
    for name in names:
        node_type = getattr(ast, name, None)
        if isinstance(node_type, type):
            found.add(node_type)
    return found


_ALLOWED_NODES: Set[type] = _node_types(
    # structure
    "Expression",
    "BinOp",
    "UnaryOp",
    "BoolOp",
    "Compare",
    "Call",
    "keyword",
    "Name",
    "Load",
    "Constant",
    "Tuple",
    "List",
    "Subscript",
    "Index",
    "Slice",
    "IfExp",
    # binary operators
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "MatMult",
    "BitAnd",
    "BitOr",
    "BitXor",
    # unary operators
    "Invert",
    "USub",
    "UAdd",
    "Not",
    # boolean operators
    "And",
    "Or",
    # comparisons
    "Eq",
    "NotEq",
    "Lt",
    "LtE",
    "Gt",
    "GtE",
)

#: Tailored messages for node types a hostile (or merely confused) user is likely to reach for.
_REJECTION_HINTS: Dict[str, str] = {
    "Attribute": "attribute access ('.') is not allowed",
    "Lambda": "lambda expressions are not allowed",
    "ListComp": "list comprehensions are not allowed",
    "SetComp": "set comprehensions are not allowed",
    "DictComp": "dict comprehensions are not allowed",
    "GeneratorExp": "generator expressions are not allowed",
    "Dict": "dict literals are not allowed",
    "Set": "set literals are not allowed",
    "Starred": "argument unpacking ('*') is not allowed",
    "JoinedStr": "f-strings are not allowed",
    "FormattedValue": "f-strings are not allowed",
    "NamedExpr": "assignment expressions (':=') are not allowed",
    "Await": "await is not allowed",
    "Yield": "yield is not allowed",
    "YieldFrom": "yield is not allowed",
    "In": "the 'in' operator is not allowed",
    "NotIn": "the 'in' operator is not allowed",
    "Is": "the 'is' operator is not allowed",
    "IsNot": "the 'is' operator is not allowed",
}


def _reject(node: ast.AST) -> "ExpressionError":
    """Build the ExpressionError for a disallowed node."""
    name = type(node).__name__
    hint = _REJECTION_HINTS.get(name)
    if hint is not None:
        return ExpressionError(hint)
    return ExpressionError(f"{name} is not allowed in an expression")


def _check_identifier(name: str, kind: str = "name") -> None:
    """Reject dunder identifiers -- defense in depth behind the Attribute ban."""
    if "__" in name:
        raise ExpressionError(f"{kind} '{name}' is not allowed (identifiers may not contain '__')")


def _check_pow_exponent(value: float) -> None:
    """Reject a statically-known ``**`` exponent big enough to be a CPU bomb."""
    if not math.isfinite(value):
        return
    if abs(value) > MAX_POW_EXPONENT:
        raise ExpressionError(
            f"exponent {value:.6g} is too large (limit {MAX_POW_EXPONENT:.0f}); "
            "this would hang the application"
        )


def _fold_number(node: ast.AST) -> Optional[float]:
    """Best-effort static evaluation of a numeric-literal subtree.

    Returns ``None`` when the subtree is not made purely of numeric literals (e.g. it
    references a variable), meaning no static bound can be proven. All arithmetic is done in
    *float*, so a hostile literal such as ``9 ** 9 ** 9`` overflows in microseconds instead
    of allocating a multi-megabyte integer inside the validator.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            return float(value)
        except (OverflowError, ValueError):
            # An integer literal too big for a float is, by definition, over every limit.
            return math.inf
    if isinstance(node, ast.UnaryOp):
        operand = _fold_number(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left = _fold_number(node.left)
        right = _fold_number(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow):
            # Check before computing: this is what stops the validator bombing itself.
            _check_pow_exponent(right)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return float(left // right)
            if isinstance(node.op, ast.Mod):
                return float(left % right)
            if isinstance(node.op, ast.Pow):
                return float(left**right)
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
    return None


def _check_node(node: ast.AST, allowed_names: Optional[Set[str]]) -> None:
    """Apply every allowlist rule to a single node. Raises ExpressionError on any violation."""
    if type(node) not in _ALLOWED_NODES:
        raise _reject(node)

    if isinstance(node, ast.Name):
        _check_identifier(node.id)
        if allowed_names is not None and node.id not in allowed_names:
            raise ExpressionError(f"unknown name '{node.id}'")
        return

    if isinstance(node, ast.keyword):
        if node.arg is None:
            raise ExpressionError("keyword unpacking ('**') is not allowed")
        _check_identifier(node.arg, kind="keyword argument")
        return

    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name):
            raise ExpressionError(
                "only direct calls to built-in functions are allowed "
                "(the thing being called must be a plain function name)"
            )
        _check_identifier(func.id)
        target = SAFE_NAMES.get(func.id)
        if target is None:
            raise ExpressionError(f"unknown function '{func.id}'")
        if not callable(target):
            raise ExpressionError(f"'{func.id}' is a constant, not a function")
        return

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, bytes)) and len(value) > MAX_STRING_CHARS:
            raise ExpressionError(
                f"string constant is too long ({len(value)} > {MAX_STRING_CHARS} characters)"
            )
        return

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = _fold_number(node.right)
        if exponent is not None:
            _check_pow_exponent(exponent)


#: The sample-index pseudo-variable. A user writes ``#``; because ``#`` starts a comment in
#: Python it is rewritten to this reserved identifier *before* the expression is ever parsed,
#: and bound to the 0-based index of each sample (``np.arange(N)``) at evaluation. So ``#``
#: always works as "the row / sample number", e.g. ``sin(0.1 * #)`` or ``y + #``.
HASH_TOKEN = "#"
HASH_NAME = "hashidx_"


def _desugar_hash(expr: str) -> str:
    """Rewrite the ``#`` sample-index token to :data:`HASH_NAME` before parsing.

    ``#`` is Python's comment character, so it must never reach ``ast.parse`` verbatim (the
    rest of the line would vanish). Math expressions have no string literals to protect, so a
    plain replace is safe.
    """
    if HASH_TOKEN not in expr:
        return expr
    return expr.replace(HASH_TOKEN, HASH_NAME)


def _uses_name(tree: ast.AST, name: str) -> bool:
    """True when ``tree`` references the bare variable ``name``."""
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(tree))


def _infer_index_length(variables: Mapping[str, Any]) -> int:
    """Length for the ``#`` index array: the longest 1-D array binding, else 1 (a scalar)."""
    best = 0
    for value in variables.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            best = max(best, int(arr.shape[0]))
    return best if best > 0 else 1


def validate(expr: str, *, names: Optional[Iterable[str]] = None) -> ast.Expression:
    """Parse ``expr`` and prove it is safe to evaluate. Returns the validated AST.

    Every node type outside the allowlist is rejected, dotted access is impossible, calls are
    restricted to :data:`SAFE_NAMES` callables, and the size/exponent bombs are bounded. See
    the module docstring for the full rationale.

    ``names`` optionally restricts bare names to :data:`SAFE_NAMES` plus the given names --
    pass it to reject unknown variables at parse time with a good message. When it is ``None``
    (the default) unknown bare names are accepted as free variables to be supplied later via
    ``evaluate(variables=...)``; they are inert until bound, because evaluation runs with empty
    builtins.

    Raises:
        ExpressionError: if the expression is empty, too long, syntactically invalid, or
            contains anything outside the allowlist.
    """
    if not isinstance(expr, str):
        raise ExpressionError(f"expression must be a string, got {type(expr).__name__}")
    if len(expr) > MAX_EXPR_CHARS:
        raise ExpressionError(f"expression is too long ({len(expr)} > {MAX_EXPR_CHARS} characters)")
    if not expr.strip():
        raise ExpressionError("expression is empty")

    # Rewrite '#' -> HASH_NAME before parsing: '#' is a comment char and cannot survive
    # ast.parse. The sample index it stands for is bound at evaluation, not here.
    expr = _desugar_hash(expr)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ExpressionError(f"could not parse expression: {exc}") from exc

    allowed_names: Optional[Set[str]] = None
    if names is not None:
        allowed_names = set(SAFE_NAMES)
        allowed_names.update(names)
        allowed_names.add(HASH_NAME)  # '#' is always a permitted (auto-bound) variable

    try:
        for node in ast.walk(tree):
            _check_node(node, allowed_names)
    except RecursionError as exc:  # pragma: no cover - needs a pathologically nested input
        raise ExpressionError("expression is nested too deeply") from exc

    return tree


def free_variables(expr: str, *, exclude: Iterable[str] = ("x",)) -> List[str]:
    """Return the sorted names ``expr`` references that are neither built in nor excluded.

    This is what the functions panel uses to auto-create a parameter slider per free variable:
    ``a * sin(b * x)`` -> ``["a", "b"]``. Validates first, so a hostile expression raises
    rather than reporting variables.
    """
    tree = validate(expr)
    # HASH_NAME is auto-bound (the '#' sample index), never a user parameter, so it must not
    # surface as a slider the way 'a'/'b' do.
    known = set(SAFE_NAMES) | set(exclude) | {HASH_NAME}
    found = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return sorted(found - known)


def _check_variable_names(variables: Mapping[str, Any]) -> None:
    """Reject caller-supplied bindings that could not have been typed by a user."""
    for key in variables:
        if not isinstance(key, str) or not key.isidentifier():
            raise ExpressionError(f"invalid variable name: {key!r}")
        _check_identifier(key, kind="variable")


def evaluate(expr: str, variables: Optional[Mapping[str, Any]] = None) -> Any:
    """Validate and evaluate ``expr``, returning whatever numpy makes of it.

    ``variables`` are bound on top of :data:`SAFE_NAMES` (so a variable named ``e`` shadows
    Euler's number). Evaluation uses empty builtins and ``np.errstate(all="ignore")``, so
    division by zero over an array yields ``inf``/``nan`` the way numpy users expect rather
    than raising; scalar ``1/0`` still raises Python's ZeroDivisionError and is reported as an
    :class:`ExpressionError`.

    Raises:
        ExpressionError: for a rejected expression, an unknown name, or any failure while
            evaluating. This function never lets another exception type escape.
    """
    bindings: Dict[str, Any] = dict(variables) if variables else {}
    _check_variable_names(bindings)

    tree = validate(expr)

    namespace: Dict[str, Any] = dict(SAFE_NAMES)
    namespace.update(bindings)
    # Bind '#' (the sample index) when the expression uses it and the caller has not supplied
    # it: length is the longest array binding, so `col + #` and `#**2` just work over a table.
    if HASH_NAME not in namespace and _uses_name(tree, HASH_NAME):
        namespace[HASH_NAME] = np.arange(_infer_index_length(bindings), dtype=np.float64)

    try:
        code = compile(tree, "<glplot>", "eval")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ExpressionError(f"could not compile expression: {exc}") from exc

    try:
        with np.errstate(all="ignore"):
            return eval(code, {"__builtins__": {}}, namespace)  # nosec B307
    except ExpressionError:
        raise
    except NameError as exc:
        raise ExpressionError(f"unknown name: {exc}") from exc
    except ZeroDivisionError as exc:
        raise ExpressionError(f"division by zero: {exc}") from exc
    except RecursionError as exc:
        raise ExpressionError("expression is nested too deeply") from exc
    except MemoryError as exc:
        raise ExpressionError("expression needs more memory than is available") from exc
    except Exception as exc:
        raise ExpressionError(f"{type(exc).__name__}: {exc}") from exc


def compile_1d(
    expr: str, *, variables: Optional[Mapping[str, Any]] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Validate ``expr`` once and return ``f(x) -> y``, for callers that evaluate repeatedly.

    :func:`evaluate_1d` parses, validates and compiles on every call, which is right for a
    one-shot "generate this curve" and wrong for anything re-evaluated continuously — a
    :class:`~glplot.core.layers.FunctionLayer` re-samples on every view change, so the
    parse would run on every frame of a drag to produce the identical code object.

    Validation still happens here, at build time, so a rejected expression raises now
    rather than from inside a render loop where nothing can report it. ``variables`` is
    snapshotted at build time for the same reason a queued command copies its options: the
    returned callable outlives this call, and a slider moved afterwards must not silently
    rewrite what an already-plotted curve means. Rebuild to pick up new values.

    Raises:
        ExpressionError: for anything :func:`validate` rejects.
    """
    bindings: Dict[str, Any] = dict(variables) if variables else {}
    _check_variable_names(bindings)
    tree = validate(expr)
    try:
        code = compile(tree, "<glplot>", "eval")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ExpressionError(f"could not compile expression: {exc}") from exc
    uses_hash = _uses_name(tree, HASH_NAME)

    def evaluate_over(x: np.ndarray) -> np.ndarray:
        xa = np.asarray(x, dtype=np.float64)
        if xa.ndim != 1:
            raise ExpressionError(f"x must be a 1-D array, got {xa.ndim}-D")
        namespace: Dict[str, Any] = dict(SAFE_NAMES)
        namespace.update(bindings)
        namespace["x"] = xa
        if uses_hash and HASH_NAME not in namespace:
            namespace[HASH_NAME] = np.arange(xa.shape[0], dtype=np.float64)
        try:
            with np.errstate(all="ignore"):
                result = eval(code, {"__builtins__": {}}, namespace)  # nosec B307
        except ExpressionError:
            raise
        except Exception as exc:
            raise ExpressionError(f"{type(exc).__name__}: {exc}") from exc
        return _as_1d_float(result, xa)

    evaluate_over.__name__ = "f"
    evaluate_over.expression = expr  # type: ignore[attr-defined]
    return evaluate_over


def _as_1d_float(result: Any, xa: np.ndarray) -> np.ndarray:
    """Coerce an expression result to a float64 array shaped like ``xa``.

    Shared by :func:`evaluate_1d` and :func:`compile_1d` so the two cannot drift on what a
    scalar, a bool array or a length mismatch means.
    """
    try:
        arr = np.asarray(result)
    except Exception as exc:
        raise ExpressionError(f"expression produced a value numpy cannot use: {exc}") from exc

    if np.iscomplexobj(arr):
        raise ExpressionError(
            "expression produced complex values; wrap it in abs(...) to plot the magnitude"
        )
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise ExpressionError(
            f"expression produced a non-numeric result (dtype {arr.dtype}); expected numbers"
        )

    arr = arr.astype(np.float64, copy=False)

    if arr.ndim == 0:
        return np.full(xa.shape, float(arr), dtype=np.float64)
    if arr.ndim != 1:
        raise ExpressionError(f"expression produced a {arr.ndim}-D result; expected 1-D")
    if arr.shape[0] != xa.shape[0]:
        if arr.shape[0] == 1:
            return np.full(xa.shape, float(arr[0]), dtype=np.float64)
        raise ExpressionError(
            f"expression produced {arr.shape[0]} values but x has {xa.shape[0]}; "
            "the result must line up with the domain"
        )

    out = np.ascontiguousarray(arr, dtype=np.float64)
    if np.may_share_memory(out, xa):
        out = out.copy()
    return out


def evaluate_1d(
    expr: str, x: np.ndarray, *, variables: Optional[Mapping[str, Any]] = None
) -> np.ndarray:
    """Evaluate ``expr`` over the 1-D domain ``x`` and return a float64 array of ``len(x)``.

    ``x`` is bound as the name ``x`` and always wins over a same-named entry in ``variables``.
    A scalar result (``"1"``) is broadcast to ``x.shape``. Boolean and integer results are
    cast to float; a complex result is an error, since the plot pipeline is real-valued.

    The returned array never aliases ``x``, so the caller may hand it straight to a layer.

    Raises:
        ExpressionError: for a rejected expression, a complex/non-numeric result, or a result
            whose length does not match ``x``.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 1:
        raise ExpressionError(f"x must be a 1-D array, got {xa.ndim}-D")

    bindings: Dict[str, Any] = dict(variables) if variables else {}
    _check_variable_names(bindings)
    bindings["x"] = xa

    result = evaluate(expr, bindings)
    return _as_1d_float(result, xa)


def evaluate_2d(
    expr: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    variables: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Evaluate ``expr`` over the grid of ``x`` by ``y``, returning a ``(len(y), len(x))`` matrix.

    The 2-D counterpart of :func:`evaluate_1d`, and it goes through the same
    :func:`evaluate` — so the same validated AST, the same rejections, the same safe
    names. ``x`` and ``y`` are bound as meshed 2-D arrays and both win over same-named
    entries in ``variables``, which is what makes ``y`` an *axis* here rather than the
    slider parameter it would otherwise become.

    Row ``i`` is ``y[i]``, column ``j`` is ``x[j]`` — the layout ``imshow`` expects, so
    the caller can hand the matrix straight over with ``extent=(x[0], x[-1], y[0], y[-1])``.

    An expression naming only ``x`` (or none at all) is not an error: it broadcasts, and
    the field is simply constant along the axis it ignores.

    Raises:
        ExpressionError: for a rejected expression, a complex/non-numeric result, or a
            result that does not broadcast to the grid.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1:
        raise ExpressionError(f"x must be a 1-D array, got {xa.ndim}-D")
    if ya.ndim != 1:
        raise ExpressionError(f"y must be a 1-D array, got {ya.ndim}-D")

    bindings: Dict[str, Any] = dict(variables) if variables else {}
    _check_variable_names(bindings)
    xx, yy = np.meshgrid(xa, ya)
    bindings["x"] = xx
    bindings["y"] = yy

    result = evaluate(expr, bindings)

    try:
        arr = np.asarray(result)
    except Exception as exc:
        raise ExpressionError(f"expression produced a value numpy cannot use: {exc}") from exc

    if np.iscomplexobj(arr):
        raise ExpressionError(
            "expression produced complex values; wrap it in abs(...) to plot the magnitude"
        )
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise ExpressionError(
            f"expression produced a non-numeric result (dtype {arr.dtype}); expected numbers"
        )

    arr = arr.astype(np.float64, copy=False)
    try:
        # Broadcast rather than demand the exact shape: `1`, `x` and `sin(x)*cos(y)` are
        # all legitimate fields, and only the last of them arrives already meshed.
        out = np.broadcast_to(arr, xx.shape)
    except ValueError as exc:
        raise ExpressionError(
            f"expression produced a result of shape {arr.shape}, which does not fit the "
            f"{xx.shape[0]} x {xx.shape[1]} grid"
        ) from exc

    # `broadcast_to` returns a read-only view over the operand; copy so the caller owns a
    # real, writable array it can hand to a layer.
    return np.ascontiguousarray(out, dtype=np.float64)
