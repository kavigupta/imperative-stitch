import ast

from parameterized import parameterized

from imperative_stitch.analyze_program.extract.errors import (
    ClosureOverVariableModifiedInExtractedCode,
)
from imperative_stitch.utils.ast_utils import (
    ReplaceNodes,
)
from imperative_stitch.utils.classify_nodes import compute_types_each
from imperative_stitch.utils.run_code import normalize_output, passes_tests, run_python

from .extract_test import GenericExtractRealisticTest, GenericExtractTest
from .utils import small_set_examples, small_set_runnable_code_examples


class RewriteTest(GenericExtractTest):
    def test_basic_rewrite(self):
        code = """
        def f(x, y):
            __start_extract__
            z = x ** 2 + {__metavariable__, __m1, y ** 2}
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x, y):
            z = __f0(x, lambda: y ** 2)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1):
            __0 = __1 ** 2 + __m1()
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_rewrite_capturing_a_variable(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 7
            z = {__metavariable__, __m1, y ** 7 - y}
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            z = __f0(x, lambda y: y ** 7 - y)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1):
            __0 = __1 ** 7
            __2 = __m1(__0)
            return __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_rewrite_capturing_a_closed_variable(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 7
            z = lambda x: {__metavariable__, __m1, y ** 7 - x}
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            z = __f0(x, lambda y, x: y ** 7 - x)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1):
            __0 = __1 ** 7
            __2 = lambda __3: __m1(__0, __3)
            return __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_multiple_rewrites(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 7
            z = lambda x: {__metavariable__, __m1, y ** 7} - {__metavariable__, __m2, x}
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            z = __f0(x, lambda y: y ** 7, lambda x: x)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1, __m2):
            __0 = __1 ** 7
            __2 = lambda __3: __m1(__0) - __m2(__3)
            return __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_rewrite_inside_comprehension(self):
        code = """
        def f(x):
            y = x ** 7
            __start_extract__
            z = [{__metavariable__, __m1, y ** 7 - x} for x in range(10)]
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            y = x ** 7
            z = __f0(lambda x: y ** 7 - x)
            return z
        """
        post_extracted = """
        def __f0(__m1):
            __0 = [__m1(__1) for __1 in range(10)]
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_rewrite_inside_comprehension(self):
        code = """
        def f(x):
            y = x ** 7
            __start_extract__
            z = [{__metavariable__, __m1, y ** 7 - x} for x in range(10)]
            y = 3
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            y = x ** 7
            z = __f0(lambda x: y ** 7 - x)
            return z
        """
        post_extracted = """
        def __f0(__m1):
            __0 = [__m1(__2) for __2 in range(10)]
            __1 = 3
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_rewrite_inside_genexpr_extracted_contains_reassign(self):
        code = """
        def f(x):
            y = x ** 7
            __start_extract__
            z = ({__metavariable__, __m1, y ** 7 - x} for x in range(10))
            y = 3
            __end_extract__
            return z
        """
        self.assertEqual(
            self.run_extract(code), ClosureOverVariableModifiedInExtractedCode()
        )

    def test_rewrite_inside_genexpr_extracted_not_contains_reassign(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 7
            z = ({__metavariable__, __m1, y ** 7 - x} for x in range(10))
            y = 3
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            z = __f0(x, lambda y, x: y ** 7 - x)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1):
            __0 = __1 ** 7
            __2 = (__m1(__0, __3) for __3 in range(10))
            __0 = 3
            return __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_in_conditional(self):
        code = """
        def _get_content_type(url, session):
            __start_extract__
            if {__metavariable__, __m1, scheme not in ('http', 'https')}:
                return ''
            return 2
            __end_extract__
        """
        post_extract_expected = """
        def _get_content_type(url, session):
            return __f0(lambda: scheme not in ('http', 'https'))
        """
        post_extracted = """
        def __f0(__m1):
            if __m1():
                return ''
            return 2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_duplicated_metavariable(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 7
            z = lambda x: {__metavariable__, __m1, y ** 7} - {__metavariable__, __m1, y ** 7}
            __end_extract__
            return z
        """
        post_extract_expected = """
        def f(x):
            z = __f0(x, lambda y: y ** 7)
            return z
        """
        post_extracted = """
        def __f0(__1, __m1):
            __0 = __1 ** 7
            __2 = lambda __3: __m1(__0) - __m1(__0)
            return __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_real_001(self):
        code = """
        def f(x, y):
            __start_extract__
            x = x ** 3
            y = x ** 7
            return y ** {__metavariable__, __m1, x - y}
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, lambda x, y: x - y)
        """
        post_extracted = """
        def __f0(__0, __m1):
            __0 = __0 ** 3
            __1 = __0 ** 7
            return __1 ** __m1(__0, __1)
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_augadd_const(self):
        code = """
        def f(x, y):
            __start_extract__
            x += {__metavariable__, __m1, 1}
            __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            x = __f0(x, lambda: 1)
            return x
        """
        post_extracted = """
        def __f0(__0, __m1):
            __0 += __m1()
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_metavariable_uses_variable_from_entry_but_origin_is_within_site(self):
        code = """
        def f(n):
            x = 1
            __start_extract__
            while x <= n:
                x = 1 + {__metavariable__, __m0, x}
            __end_extract__
            return x
        """
        post_extract_expected = """
        def f(n):
            x = 1
            x = __f0(x, n, lambda x: x)
            return x
        """
        post_extracted = """
        def __f0(__0, __1, __m0):
            while __0 <= __1:
                __0 = 1 + __m0(__0)
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )


class GenericRewriteRealisticTest(GenericExtractRealisticTest):
    def get_expressions(self, body, start="S"):
        expressions = list(
            x for x, state in compute_types_each(body, start) if state == "E"
        )
        expressions = [x for x in expressions if not self.bad_expression(x)]
        return expressions

    def bad_expression(self, expr, top_level=True):
        if top_level and isinstance(expr, ast.Starred):
            return True
        if isinstance(expr, (ast.Slice, ast.Ellipsis)):
            return True
        if isinstance(expr, ast.Tuple):
            return any(self.bad_expression(x, top_level=False) for x in expr.elts)
        return False

    def manipulate(self, body, rng):
        expressions = self.get_expressions(body)
        if not expressions:
            return body, 0
        count = rng.randint(1, min(5, len(expressions) + 1))
        expressions = sample_non_overlapping(expressions, count, rng)
        print([repr(ast.unparse(e)) for e in expressions])
        replace = ReplaceNodes(
            {
                expr: ast.Set(
                    elts=[ast.Name("__metavariable__"), ast.Name(f"__m{i}"), expr]
                )
                for i, expr in enumerate(expressions)
            }
        )
        body = [replace.visit(stmt) for stmt in body]
        return body, len(expressions)


class RewriteRealisticTest(GenericRewriteRealisticTest):
    @parameterized.expand([(i,) for i in range(len(small_set_examples()))])
    def test_realistic(self, i):
        self.operate(i)


class RewriteSemanticsTest(GenericRewriteRealisticTest):
    @parameterized.expand(
        [(i,) for i in range(len(small_set_runnable_code_examples()))]
    )
    def test_semantics(self, i):
        example = small_set_runnable_code_examples()[i]
        code = example["solution"]
        xs = self.operate_on_code(i, code, use_full_tree=True)
        for code, out in xs:
            if isinstance(out, Exception):
                continue
            post_extract, extracted = out
            new_code = "\n".join([extracted, post_extract])
            print("*" * 80)
            print(code)
            print("=" * 80)
            print(new_code)
            for input, output in zip(example["inputs"], example["outputs"]):
                py_out = run_python(new_code, input)
                if py_out is None:
                    run_python.function(new_code, input)
                    self.fail("error")
                py_out = normalize_output(py_out)
                output = normalize_output(output)
                self.assertEqual(py_out, output)


def sample_non_overlapping(xs, count, rng):
    result = []
    while xs:
        i = rng.randint(len(xs))
        result.append(xs[i])
        xs = xs[:i] + xs[i + 1 :]
        if len(result) == count:
            break
        xs = [x for x in xs if not overlapping(x, result[-1])]
    return result


def overlapping(x, y):
    return set(ast.walk(x)) & set(ast.walk(y))