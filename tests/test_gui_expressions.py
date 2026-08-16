"""Test the AST-validated expression evaluator in glplot.gui.expressions.

This module is the security boundary of the GUI's function generator: it is the only place
where user-typed text becomes executable code. The tests below are therefore weighted towards
proving that hostile input is rejected at *parse* time and never executes at all.

No OpenGL context, no window and no GUI toolkit are required -- the evaluator is pure numpy.
"""

from __future__ import annotations

import builtins
import os

import numpy as np
import pytest

from glplot.gui.expressions import (
    MAX_EXPR_CHARS,
    MAX_NOISE_SAMPLES,
    MAX_POW_EXPONENT,
    MAX_STRING_CHARS,
    SAFE_NAMES,
    ExpressionError,
    evaluate,
    evaluate_1d,
    free_variables,
    validate,
)

#: Expressions that must never be accepted, keyed by a short label for readable test ids.
HOSTILE_EXPRESSIONS = {
    "import_system": '__import__("os").system("touch /tmp/pwned")',
    "subclasses_escape": "().__class__.__bases__[0].__subclasses__()",
    "class_attribute": "x.__class__",
    "open_passwd": 'open("/etc/passwd")',
    "eval_call": 'eval("1")',
    "exec_call": 'exec("import os")',
    "lambda_expr": "lambda x: x",
    "list_comp": "[i for i in x]",
    "set_comp": "{i for i in x}",
    "dict_comp": "{i: i for i in x}",
    "generator_exp": "(i for i in x)",
    "walrus": "(y := 1)",
    "starred_args": "sin(*x)",
    "kwargs_unpacking": "sin(**x)",
    "fstring": 'f"{x}"',
    "cpu_bomb": "9**9**9",
    "attribute_access": "x.real",
    "method_call": "x.mean()",
    "dunder_name": "__builtins__",
    "dunder_keyword": "sin(__x__=1)",
    "dict_literal": "{1: 2}",
    "set_literal": "{1, 2}",
    "print_call": 'print("hi")',
    "globals_call": "globals()",
    "locals_call": "locals()",
    "vars_call": "vars()",
    "getattr_call": 'getattr(x, "__class__")',
    "type_call": "type(x)",
    "dir_call": "dir()",
    "compile_call": 'compile("1", "<s>", "eval")',
    "breakpoint_call": "breakpoint()",
    "call_of_call": "sin(x)(1)",
    "call_of_subscript": "x[0](1)",
    "in_operator": "1 in x",
    "is_operator": "x is x",
}


class TestHostileExpressionsAreRejected:
    """Every known sandbox-escape shape must raise ExpressionError."""

    @pytest.mark.parametrize("expr", HOSTILE_EXPRESSIONS.values(), ids=HOSTILE_EXPRESSIONS.keys())
    def test_validate_rejects(self, expr):
        """Test that validate() rejects each hostile expression."""
        with pytest.raises(ExpressionError):
            validate(expr)

    @pytest.mark.parametrize("expr", HOSTILE_EXPRESSIONS.values(), ids=HOSTILE_EXPRESSIONS.keys())
    def test_evaluate_rejects(self, expr):
        """Test that evaluate() rejects each hostile expression with x bound."""
        with pytest.raises(ExpressionError):
            evaluate(expr, {"x": np.linspace(0.0, 1.0, 4)})

    @pytest.mark.parametrize("expr", HOSTILE_EXPRESSIONS.values(), ids=HOSTILE_EXPRESSIONS.keys())
    def test_evaluate_1d_rejects(self, expr):
        """Test that evaluate_1d() rejects each hostile expression."""
        with pytest.raises(ExpressionError):
            evaluate_1d(expr, np.linspace(0.0, 1.0, 4))

    def test_rejection_messages_are_human_readable(self):
        """Test that rejection messages are lowercase prose, not tracebacks or node dumps."""
        for expr in HOSTILE_EXPRESSIONS.values():
            with pytest.raises(ExpressionError) as info:
                validate(expr)
            message = str(info.value)
            assert message, f"empty message for {expr!r}"
            assert "Traceback" not in message
            assert "<ast." not in message
            assert len(message) < 200


class TestHostileExpressionsDoNotExecute:
    """Prove non-execution with sentinels: a rejected expression cannot touch them."""

    def test_import_system_never_runs(self, monkeypatch):
        """Test that __import__("os").system(...) never reaches os.system."""
        systemed = []
        imported = []
        real_import = builtins.__import__

        def spy_import(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(os, "system", lambda cmd: systemed.append(cmd) or 0)
        monkeypatch.setattr(builtins, "__import__", spy_import)
        with pytest.raises(ExpressionError):
            evaluate('__import__("os").system("touch /tmp/pwned")')
        monkeypatch.undo()
        assert systemed == []
        assert "os" not in imported

    def test_open_never_runs(self, monkeypatch):
        """Test that open("/etc/passwd") never reaches builtins.open."""
        fired = []
        monkeypatch.setattr(builtins, "open", lambda *a, **k: fired.append(a) or None)
        with pytest.raises(ExpressionError):
            evaluate('open("/etc/passwd")')
        with pytest.raises(ExpressionError):
            evaluate_1d('open("/etc/passwd")', np.zeros(3))
        assert fired == []

    def test_eval_and_exec_never_run(self, monkeypatch):
        """Test that eval()/exec() in user text never reach the real builtins."""
        fired = []
        real_eval, real_exec = builtins.eval, builtins.exec
        monkeypatch.setattr(builtins, "eval", lambda *a, **k: fired.append(a) or real_eval(*a, **k))
        monkeypatch.setattr(builtins, "exec", lambda *a, **k: fired.append(a) or real_exec(*a, **k))
        for expr in ('eval("1")', 'exec("x = 1")'):
            with pytest.raises(ExpressionError):
                evaluate(expr)
        monkeypatch.undo()
        assert fired == []

    def test_bound_callable_variable_is_never_called(self):
        """Test that a callable bound as a variable cannot be invoked -- calls need SAFE_NAMES."""
        fired = []

        def boom(*args):
            fired.append(args)
            return 1.0

        with pytest.raises(ExpressionError):
            evaluate("boom(1)", {"boom": boom})
        assert fired == []

    def test_cpu_bomb_is_rejected_without_computing(self):
        """Test that 9**9**9 is rejected quickly, without the validator materialising a bignum."""
        with pytest.raises(ExpressionError) as info:
            validate("9**9**9")
        assert "exponent" in str(info.value)
        assert "too large" in str(info.value)

    def test_builtins_are_empty_during_evaluation(self):
        """Test that a non-dunder builtin name is unreachable, proving builtins are stripped."""
        with pytest.raises(ExpressionError) as info:
            evaluate("len")
        assert "unknown name" in str(info.value)


class TestValidate:
    """Test validate() acceptance, the strict names allowlist, and input guards."""

    @pytest.mark.parametrize(
        "expr",
        [
            "sin(x)",
            "a * sin(b * x) + noise(x, 0.1)",
            "gauss(x, mu=0, sigma=1)",
            "x if x > 0 else -x",
            "(x > 0) & (x < 1)",
            "clip(x, -1, 1)",
            "x[0]",
            "x[1:3]",
            "array([1.0, 2.0])",
            "-x ** 2",
            "x // 2 % 3",
            "not x",
            "pi + e + tau",
        ],
    )
    def test_accepts_legitimate_expressions(self, expr):
        """Test that ordinary maths expressions validate cleanly."""
        assert validate(expr) is not None

    def test_unknown_bare_name_allowed_by_default(self):
        """Test that unknown bare names pass as free variables when names=None."""
        assert validate("a * sin(b * x)") is not None

    def test_names_allowlist_rejects_unknown_name(self):
        """Test that passing names=(...) makes unknown bare names a parse-time error."""
        with pytest.raises(ExpressionError) as info:
            validate("a * q", names=("a",))
        assert "unknown name 'q'" in str(info.value)

    def test_names_allowlist_still_permits_safe_names(self):
        """Test that SAFE_NAMES remain usable when a names allowlist is supplied."""
        assert validate("sin(a) + pi", names=("a",)) is not None

    def test_empty_expression_rejected(self):
        """Test that an empty or whitespace-only expression is rejected."""
        for expr in ("", "   ", "\n\t"):
            with pytest.raises(ExpressionError) as info:
                validate(expr)
            assert "empty" in str(info.value)

    def test_non_string_rejected(self):
        """Test that a non-string expression is rejected with its type named."""
        with pytest.raises(ExpressionError) as info:
            validate(123)
        assert "must be a string" in str(info.value)
        assert "int" in str(info.value)

    def test_too_long_expression_rejected(self):
        """Test that an over-long expression is rejected before parsing."""
        with pytest.raises(ExpressionError) as info:
            validate("1+" * MAX_EXPR_CHARS)
        assert "too long" in str(info.value)

    def test_long_string_constant_rejected(self):
        """Test that an over-long string constant is rejected."""
        expr = '"' + "a" * (MAX_STRING_CHARS + 1) + '"'
        with pytest.raises(ExpressionError) as info:
            validate(expr)
        assert "string constant is too long" in str(info.value)

    def test_syntax_error_is_wrapped(self):
        """Test that a syntax error becomes a readable ExpressionError."""
        with pytest.raises(ExpressionError) as info:
            validate("sin(")
        assert "syntax error" in str(info.value)

    def test_statement_is_rejected(self):
        """Test that a statement (not an expression) is a syntax error, not an assignment."""
        with pytest.raises(ExpressionError):
            validate("y = 1")

    def test_constant_call_rejected(self):
        """Test that calling a SAFE_NAMES constant reports it as a constant."""
        with pytest.raises(ExpressionError) as info:
            validate("pi(2)")
        assert "'pi' is a constant, not a function" in str(info.value)

    def test_pow_exponent_limit(self):
        """Test that a statically-known exponent above MAX_POW_EXPONENT is rejected."""
        with pytest.raises(ExpressionError):
            validate(f"x ** {MAX_POW_EXPONENT * 10:.0f}")
        assert validate("x ** 2") is not None

    def test_pow_with_variable_exponent_allowed(self):
        """Test that a non-static exponent cannot be bounded and is therefore allowed."""
        assert validate("x ** a") is not None


class TestFreeVariables:
    """Test free_variables()."""

    def test_reports_sorted_free_variables(self):
        """Test that free variables are returned sorted, excluding x and SAFE_NAMES."""
        assert free_variables("a * sin(b * x) + pi") == ["a", "b"]

    def test_no_free_variables(self):
        """Test that an expression of only builtins and x has no free variables."""
        assert free_variables("sin(x) * cos(x)") == []

    def test_exclude_is_configurable(self):
        """Test that the exclude set can be changed."""
        assert free_variables("a + t", exclude=("t",)) == ["a"]
        assert free_variables("a + x", exclude=()) == ["a", "x"]

    def test_hostile_expression_raises_instead_of_reporting(self):
        """Test that free_variables validates first and refuses hostile input."""
        with pytest.raises(ExpressionError):
            free_variables("a + x.__class__")


class TestSafeNames:
    """Test the SAFE_NAMES namespace."""

    def test_no_dunder_or_module_objects(self):
        """Test that SAFE_NAMES holds only plain identifiers and no module objects."""
        for name, value in SAFE_NAMES.items():
            assert "__" not in name
            assert name.isidentifier()
            assert not isinstance(value, type(np))

    def test_constants_have_expected_values(self):
        """Test the numeric constants."""
        assert np.allclose(SAFE_NAMES["pi"], np.pi)
        assert np.allclose(SAFE_NAMES["e"], np.e)
        assert np.allclose(SAFE_NAMES["tau"], 2.0 * np.pi)
        assert np.isinf(SAFE_NAMES["inf"])
        assert np.isnan(SAFE_NAMES["nan"])

    def test_every_name_is_reachable_from_an_expression(self):
        """Test that each SAFE_NAMES entry resolves when referenced by name."""
        for name in SAFE_NAMES:
            result = evaluate(name)
            assert result is SAFE_NAMES[name] or result == SAFE_NAMES[name] or np.isnan(result)

    def test_every_callable_name_is_callable(self):
        """Test that every non-constant SAFE_NAMES entry is callable."""
        constants = {"pi", "e", "tau", "inf", "nan"}
        for name, value in SAFE_NAMES.items():
            if name in constants:
                assert isinstance(value, float)
            else:
                assert callable(value), f"{name} is neither a constant nor callable"

    @pytest.mark.parametrize(
        "expr",
        [
            "sin(x)",
            "cos(x)",
            "tan(x)",
            "arcsin(clip(x, -1, 1))",
            "arccos(clip(x, -1, 1))",
            "arctan(x)",
            "arctan2(x, 1)",
            "sinh(x)",
            "cosh(x)",
            "tanh(x)",
            "arcsinh(x)",
            "arccosh(x + 2)",
            "arctanh(clip(x, -0.9, 0.9))",
            "exp(x)",
            "expm1(x)",
            "log(abs(x) + 1)",
            "log2(abs(x) + 1)",
            "log10(abs(x) + 1)",
            "log1p(abs(x))",
            "sqrt(abs(x))",
            "cbrt(x)",
            "power(abs(x), 2)",
            "abs(x)",
            "fabs(x)",
            "sign(x)",
            "floor(x)",
            "ceil(x)",
            "round(x)",
            "trunc(x)",
            "clip(x, 0, 1)",
            "mod(x, 2)",
            "sum(x)",
            "cumsum(x)",
            "prod(x)",
            "cumprod(x)",
            "mean(x)",
            "median(x)",
            "std(x)",
            "var(x)",
            "min(x)",
            "max(x)",
            "argmin(x)",
            "argmax(x)",
            "ptp(x)",
            "linspace(0, 1, 8)",
            "arange(8)",
            "zeros(8)",
            "ones(8)",
            "full(8, 3.0)",
            "array([1.0, 2.0])",
            "where(x > 0, x, -x)",
            "maximum(x, 0)",
            "minimum(x, 0)",
            "heaviside(x)",
            "sinc(x)",
            "gradient(x)",
            "diff(x)",
            "interp(x, linspace(-1, 1, 8), x)",
            "gauss(x)",
            "sigmoid(x)",
            "rect(x)",
            "step(x)",
            "sawtooth(x)",
            "square(x)",
            "noise(x, 0.1, 3)",
            "pi * x",
            "e ** x",
            "tau * x",
            "x + inf",
            "x + nan",
        ],
    )
    def test_name_is_usable_in_an_expression(self, expr):
        """Test that each SAFE_NAMES entry actually works when used in an expression."""
        result = evaluate(expr, {"x": np.linspace(-1.0, 1.0, 8)})
        assert result is not None


class TestConvenienceHelpers:
    """Test the maths of the helpers written for this module."""

    def test_gauss_peaks_at_one_and_is_symmetric(self):
        """Test that gauss() peaks at exactly 1.0 at mu and is symmetric about it."""
        x = np.array([-1.0, 0.0, 1.0])
        y = evaluate_1d("gauss(x)", x)
        assert np.allclose(y[1], 1.0)
        assert np.allclose(y[0], y[2])
        assert np.allclose(y[0], np.exp(-0.5))

    def test_gauss_honours_mu_and_sigma_keywords(self):
        """Test that gauss(x, mu=..., sigma=...) shifts and scales the bell."""
        x = np.array([2.0, 4.0])
        y = evaluate_1d("gauss(x, mu=2, sigma=2)", x)
        assert np.allclose(y, [1.0, np.exp(-0.5)])

    def test_gauss_zero_sigma_is_a_clean_error(self):
        """Test that gauss() with sigma=0 raises ExpressionError, not ZeroDivisionError."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("gauss(x, sigma=0)", np.zeros(3))
        assert "sigma must be non-zero" in str(info.value)

    def test_sigmoid_values(self):
        """Test sigmoid() against 1 / (1 + exp(-x))."""
        x = np.array([-2.0, 0.0, 2.0])
        y = evaluate_1d("sigmoid(x)", x)
        assert np.allclose(y, 1.0 / (1.0 + np.exp(-x)))
        assert np.allclose(y[1], 0.5)

    def test_sigmoid_does_not_overflow(self):
        """Test that sigmoid() stays finite and in (0, 1) for extreme inputs."""
        x = np.array([-1e4, 1e4])
        y = evaluate_1d("sigmoid(x)", x)
        assert np.all(np.isfinite(y))
        assert np.allclose(y, [0.0, 1.0])

    def test_rect_edges_and_shape(self):
        """Test rect(): 1 inside, 0.5 exactly on the edges, 0 outside."""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = evaluate_1d("rect(x)", x)
        assert y.shape == x.shape
        assert np.allclose(y, [0.0, 0.5, 1.0, 0.5, 0.0])

    def test_step_is_one_at_zero(self):
        """Test step(): 0 below zero, 1 at and above zero."""
        x = np.array([-1.0, -1e-9, 0.0, 1e-9, 1.0])
        y = evaluate_1d("step(x)", x)
        assert np.allclose(y, [0.0, 0.0, 1.0, 1.0, 1.0])

    def test_heaviside_defaults_to_half_at_zero(self):
        """Test that the wrapped heaviside() defaults x2 to 0.5 so heaviside(x) is typeable."""
        y = evaluate_1d("heaviside(x)", np.array([-1.0, 0.0, 1.0]))
        assert np.allclose(y, [0.0, 0.5, 1.0])

    def test_heaviside_x2_is_overridable(self):
        """Test that heaviside()'s value at zero can still be given explicitly."""
        y = evaluate_1d("heaviside(x, 1.0)", np.array([0.0]))
        assert np.allclose(y, [1.0])

    def test_sawtooth_shape_and_range(self):
        """Test sawtooth(): rises through zero and stays within [-1, 1)."""
        x = np.linspace(-2.0, 2.0, 101)
        y = evaluate_1d("sawtooth(x)", x)
        assert y.shape == x.shape
        assert np.all(y >= -1.0) and np.all(y < 1.0)
        assert np.allclose(evaluate_1d("sawtooth(x)", np.array([0.0, 0.25])), [0.0, 0.5])

    def test_sawtooth_period(self):
        """Test that sawtooth() repeats every `period`."""
        x = np.array([0.3, 2.3, 4.3])
        y = evaluate_1d("sawtooth(x, 2)", x)
        assert np.allclose(y, y[0])

    def test_sawtooth_zero_period_is_a_clean_error(self):
        """Test that sawtooth() with period=0 raises ExpressionError."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("sawtooth(x, 0)", np.zeros(3))
        assert "period must be non-zero" in str(info.value)

    def test_square_takes_only_two_values(self):
        """Test square(): +1 over the first half of each period, -1 over the second."""
        x = np.array([0.0, 0.25, 0.49, 0.5, 0.75, 0.99])
        y = evaluate_1d("square(x)", x)
        assert np.allclose(y, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        assert set(np.unique(evaluate_1d("square(x)", np.linspace(-3, 3, 97)))) == {-1.0, 1.0}

    def test_square_period(self):
        """Test that square() honours a non-unit period."""
        y = evaluate_1d("square(x, 4)", np.array([0.0, 1.9, 2.1, 3.9]))
        assert np.allclose(y, [1.0, 1.0, -1.0, -1.0])

    def test_square_zero_period_is_a_clean_error(self):
        """Test that square() with period=0 raises ExpressionError."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("square(x, 0)", np.zeros(3))
        assert "period must be non-zero" in str(info.value)

    def test_noise_matches_array_shape(self):
        """Test that noise(x, ...) returns an array shaped like x."""
        x = np.linspace(0.0, 1.0, 17)
        y = evaluate_1d("noise(x, 0.1, 1234)", x)
        assert y.shape == x.shape

    def test_noise_count_form(self):
        """Test that noise(n) allocates n samples."""
        y = evaluate("noise(64, 1.0, 0)")
        assert y.shape == (64,)

    def test_noise_seed_is_reproducible(self):
        """Test that the same seed gives the same draw and a different seed does not."""
        a = evaluate("noise(32, 1.0, 7)")
        b = evaluate("noise(32, 1.0, 7)")
        c = evaluate("noise(32, 1.0, 8)")
        assert np.allclose(a, b)
        assert not np.allclose(a, c)

    def test_noise_scale_is_the_standard_deviation(self):
        """Test that noise() scales as its `scale` argument says."""
        small = evaluate("noise(20000, 0.1, 5)")
        assert np.allclose(np.std(small), 0.1, atol=0.01)
        assert np.allclose(np.mean(small), 0.0, atol=0.01)

    def test_noise_rejects_oversized_request(self):
        """Test that noise() refuses to allocate more than MAX_NOISE_SAMPLES."""
        with pytest.raises(ExpressionError) as info:
            evaluate(f"noise({MAX_NOISE_SAMPLES + 1})")
        assert "exceeds the limit" in str(info.value)

    def test_noise_rejects_negative_count(self):
        """Test that noise() refuses a negative sample count."""
        with pytest.raises(ExpressionError) as info:
            evaluate("noise(-1)")
        assert "non-negative" in str(info.value)


class TestEvaluate:
    """Test evaluate() binding, error wrapping, and numeric behaviour."""

    def test_arithmetic(self):
        """Test that plain arithmetic evaluates."""
        assert np.allclose(evaluate("2 + 3 * 4"), 14.0)

    def test_variables_are_bound(self):
        """Test that caller-supplied variables are visible to the expression."""
        assert np.allclose(evaluate("a * b", {"a": 3.0, "b": 4.0}), 12.0)

    def test_variable_shadows_safe_name(self):
        """Test that a variable named `e` shadows Euler's number."""
        assert np.allclose(evaluate("e", {"e": 42.0}), 42.0)

    def test_unknown_name_is_wrapped(self):
        """Test that an unbound free variable becomes a readable ExpressionError."""
        with pytest.raises(ExpressionError) as info:
            evaluate("a + 1")
        assert "unknown name" in str(info.value)
        assert "'a'" in str(info.value)

    def test_dunder_variable_name_rejected(self):
        """Test that a caller cannot smuggle a dunder binding into the namespace."""
        with pytest.raises(ExpressionError) as info:
            evaluate("1", {"__builtins__": {"eval": eval}})
        assert "not allowed" in str(info.value)

    def test_non_identifier_variable_name_rejected(self):
        """Test that a non-identifier variable key is rejected."""
        with pytest.raises(ExpressionError) as info:
            evaluate("1", {"not a name": 1})
        assert "invalid variable name" in str(info.value)

    def test_scalar_division_by_zero_is_wrapped(self):
        """Test that Python's scalar 1/0 is reported as an ExpressionError."""
        with pytest.raises(ExpressionError) as info:
            evaluate("1 / 0")
        assert "division by zero" in str(info.value)

    def test_array_division_by_zero_yields_inf(self):
        """Test that array division by zero yields inf rather than raising, via np.errstate."""
        result = evaluate("1 / x", {"x": np.array([0.0, 1.0, 2.0])})
        assert np.isinf(result[0])
        assert np.allclose(result[1:], [1.0, 0.5])

    def test_invalid_operations_yield_nan_not_warnings(self):
        """Test that 0/0 and log(0) produce nan/-inf under np.errstate(all='ignore')."""
        result = evaluate("x / x", {"x": np.array([0.0, 2.0])})
        assert np.isnan(result[0])
        assert np.allclose(result[1], 1.0)
        assert np.isneginf(evaluate("log(x)", {"x": np.array([0.0])})[0])

    def test_runtime_error_is_wrapped_as_expression_error(self):
        """Test that a numpy runtime failure never escapes as its own exception type."""
        with pytest.raises(ExpressionError) as info:
            evaluate("sin(x, 1, 2, 3)", {"x": np.zeros(3)})
        assert str(info.value)

    def test_shape_mismatch_is_wrapped(self):
        """Test that a numpy broadcast failure becomes an ExpressionError."""
        with pytest.raises(ExpressionError):
            evaluate("a + b", {"a": np.zeros(3), "b": np.zeros(4)})


class TestEvaluate1d:
    """Test evaluate_1d() domain binding, broadcasting and result validation."""

    def test_returns_float64_of_len_x(self):
        """Test that the result is float64 and the same length as x."""
        x = np.linspace(0.0, 1.0, 11)
        y = evaluate_1d("sin(x)", x)
        assert y.dtype == np.float64
        assert y.shape == x.shape
        assert np.allclose(y, np.sin(x))

    def test_scalar_result_is_broadcast(self):
        """Test that a scalar result is broadcast to x.shape."""
        x = np.linspace(0.0, 1.0, 7)
        y = evaluate_1d("1", x)
        assert y.shape == x.shape
        assert np.allclose(y, 1.0)

    def test_reduction_result_is_broadcast(self):
        """Test that a reduction such as mean(x) fills the whole domain."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = evaluate_1d("mean(x)", x)
        assert y.shape == x.shape
        assert np.allclose(y, 1.5)

    def test_length_one_result_is_broadcast(self):
        """Test that a length-1 array result is broadcast to x.shape."""
        x = np.zeros(5)
        y = evaluate_1d("array([2.5])", x)
        assert np.allclose(y, np.full(5, 2.5))

    def test_length_mismatch_is_rejected(self):
        """Test that a result of the wrong length is rejected with both lengths named."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("array([1.0, 2.0])", np.zeros(5))
        message = str(info.value)
        assert "2 values" in message
        assert "x has 5" in message

    def test_complex_result_is_rejected_with_a_hint(self):
        """Test that a complex result is rejected and the message suggests abs(...)."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("x + 1j", np.zeros(3))
        message = str(info.value)
        assert "complex" in message
        assert "abs(" in message

    def test_complex_magnitude_is_accepted(self):
        """Test that the suggested abs(...) workaround actually works."""
        y = evaluate_1d("abs(x + 1j)", np.zeros(3))
        assert np.allclose(y, 1.0)

    def test_non_numeric_result_is_rejected(self):
        """Test that a string result is rejected with the dtype named."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d('array("abc")', np.zeros(3))
        assert "non-numeric" in str(info.value)

    def test_boolean_result_is_cast_to_float(self):
        """Test that a boolean mask becomes 0.0/1.0 floats."""
        x = np.array([-1.0, 0.0, 1.0])
        y = evaluate_1d("x > 0", x)
        assert y.dtype == np.float64
        assert np.allclose(y, [0.0, 0.0, 1.0])

    def test_integer_result_is_cast_to_float(self):
        """Test that an integer result becomes float64."""
        y = evaluate_1d("arange(4)", np.zeros(4))
        assert y.dtype == np.float64
        assert np.allclose(y, [0.0, 1.0, 2.0, 3.0])

    def test_multi_dimensional_result_is_rejected(self):
        """Test that a 2-D result is rejected."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("array([[1.0, 2.0], [3.0, 4.0]])", np.zeros(2))
        assert "2-D result" in str(info.value)

    def test_non_1d_domain_is_rejected(self):
        """Test that a 2-D x is rejected before evaluation."""
        with pytest.raises(ExpressionError) as info:
            evaluate_1d("x", np.zeros((2, 2)))
        assert "x must be a 1-D array" in str(info.value)

    def test_x_wins_over_a_same_named_variable(self):
        """Test that the domain x always overrides a variable also called x."""
        x = np.array([1.0, 2.0, 3.0])
        y = evaluate_1d("x", x, variables={"x": np.array([9.0, 9.0, 9.0])})
        assert np.allclose(y, x)

    def test_result_never_aliases_x(self):
        """Test that the returned array is a copy, safe to hand to a layer."""
        x = np.array([1.0, 2.0, 3.0])
        y = evaluate_1d("x", x)
        assert not np.may_share_memory(y, x)
        y[0] = 99.0
        assert np.allclose(x, [1.0, 2.0, 3.0])

    def test_extra_variables_are_bound(self):
        """Test that parameter sliders (extra variables) reach the expression."""
        x = np.linspace(0.0, 1.0, 5)
        y = evaluate_1d("a * sin(b * x)", x, variables={"a": 2.0, "b": 3.0})
        assert np.allclose(y, 2.0 * np.sin(3.0 * x))

    def test_division_by_zero_yields_inf(self):
        """Test that 1/x over a domain containing 0 yields inf, not an exception."""
        x = np.array([-1.0, 0.0, 1.0])
        y = evaluate_1d("1 / x", x)
        assert np.allclose(y[[0, 2]], [-1.0, 1.0])
        assert np.isinf(y[1])

    def test_empty_domain(self):
        """Test that an empty domain yields an empty result."""
        y = evaluate_1d("sin(x)", np.zeros(0))
        assert y.shape == (0,)

    def test_integer_domain_is_accepted(self):
        """Test that an integer x is converted to float64."""
        y = evaluate_1d("x * 2", np.arange(4))
        assert y.dtype == np.float64
        assert np.allclose(y, [0.0, 2.0, 4.0, 6.0])

    def test_list_domain_is_accepted(self):
        """Test that a plain list works as the domain."""
        y = evaluate_1d("x + 1", [0.0, 1.0, 2.0])
        assert np.allclose(y, [1.0, 2.0, 3.0])

    def test_hostile_variable_name_rejected(self):
        """Test that evaluate_1d checks caller variable names too."""
        with pytest.raises(ExpressionError):
            evaluate_1d("1", np.zeros(3), variables={"__evil__": 1})
