import ast
import unittest

import neurosym as ns
import numpy as np
from python_graphs import control_flow

from imperative_stitch.analyze_program.extract import NotApplicable, do_extract
from imperative_stitch.analyze_program.extract.errors import (
    BothYieldsAndReturns,
    ClosedVariablePassedDirectly,
    ClosureOverVariableModifiedInExtractedCode,
    ModifiesVariableClosedOverInNonExtractedCode,
    MultipleExits,
    NonInitializedInputsOrOutputs,
)
from imperative_stitch.analyze_program.extract.extract_configuration import (
    ExtractConfiguration,
)
from imperative_stitch.analyze_program.ssa.banned_component import BannedComponentError
from imperative_stitch.data import parse_extract_pragma
from imperative_stitch.utils.ast_utils import ast_nodes_in_order
from tests.utils import canonicalize, expand_with_slow_tests, small_set_examples


class GenericExtractTest(unittest.TestCase):
    def run_extract(
        self, code, num_metavariables=None, config=ExtractConfiguration(True)
    ):
        code = canonicalize(code)
        tree, [site] = parse_extract_pragma(code)
        if num_metavariables is not None:
            self.assertEqual(len(site.metavariables), num_metavariables)
        return self.run_extract_from_tree(tree, site, config=config)

    def run_extract_from_tree(self, tree, site, *, config):
        # without pragmas
        code = ast.unparse(tree)
        try:
            extr = do_extract(site, tree, extract_name="__f0", config=config)
        except (NotApplicable, BannedComponentError) as e:
            # ensure that the code is not changed
            print(type(e), e)
            self.maxDiff = None
            self.assertEqual(code, ast.unparse(tree))
            return e

        post_extract, extracted = ast.unparse(tree), ast.unparse(extr.func_def)
        extr.undo()
        self.assertEqual(code, ast.unparse(tree), "undo")
        return post_extract, extracted

    def assertCodes(self, expected, actual):
        self.maxDiff = None
        post_extract, extracted = actual
        expected_post_extract, expected_extracted = expected

        post_extract, extracted = canonicalize(post_extract), canonicalize(extracted)
        expected_post_extract, expected_extracted = (
            canonicalize(expected_post_extract),
            canonicalize(expected_extracted),
        )

        expected = expected_post_extract + "\n" + "=" * 80 + "\n" + expected_extracted
        actual = post_extract + "\n" + "=" * 80 + "\n" + extracted

        self.assertEqual(expected, actual)


class ExtractTest(GenericExtractTest):
    def test_pass(self):
        code = """
        def f(x):
            __start_extract__
            pass
            __end_extract__
        """
        post_extract_expected = """
        def f(x):
            return __f0()
        """
        post_extracted = """
        def __f0():
            pass
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_basic(self):
        code = """
        def f(x, y):
            __start_extract__
            x, y = y, x
            z = x + y
            if z > 0:
                x += 1
            __end_extract__
            return x, y
        """
        post_extract_expected = """
        def f(x, y):
            x, y = __f0(x, y)
            return x, y
        """
        post_extracted = """
        def __f0(__0, __1):
            __0, __1 = __1, __0
            __2 = __0 + __1
            if __2 > 0:
                __0 += 1
            return __0, __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_no_return(self):
        code = """
        def f(x, y):
            __start_extract__
            print(x, y)
            print(x ** 2)
            __end_extract__
            z = x + y
            if z > 0:
                x += 1
            return x, y
        """
        post_extract_expected = """
        def f(x, y):
            __f0(x, y)
            z = x + y
            if z > 0:
                x += 1
            return x, y
        """
        post_extracted = """
        def __f0(__0, __1):
            print(__0, __1)
            print(__0 ** 2)
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_basic_in_loop(self):
        code = """
        def f(xs, ys):
            zs = []
            for x, y in zip(xs, ys):
                __start_extract__
                x2y2 = x ** 2 + y ** 2
                r = x2y2 ** 0.5
                z = x + r
                __end_extract__
                zs.append(z)
            return zs
        """
        post_extract_expected = """
        def f(xs, ys):
            zs = []
            for x, y in zip(xs, ys):
                z = __f0(x, y)
                zs.append(z)
            return zs
        """
        post_extracted = """
        def __f0(__1, __2):
            __0 = __1 ** 2 + __2 ** 2
            __3 = __0 ** 0.5
            __4 = __1 + __3
            return __4
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_only_affects_loop(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                x = x + 1
                __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
            return x
        """
        post_extracted = """
        def __f0(__0):
            __0 = __0 + 1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_mutiple_returns(self):
        code = """
        def f(x, y):
            __start_extract__
            if x > 0:
                return x
            else:
                return y
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, y)
        """
        post_extracted = """
        def __f0(__0, __1):
            if __0 > 0:
                return __0
            else:
                return __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def return_at_end_by_default(self):
        code = """
        def f(x, y):
            __start_extract__
            if x > 0:
                return x
            y += 1
            return y
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, y)
        """
        post_extracted = """
        def __f0(__0, __1):
            if __0 > 0:
                return __0
            __1 += 1
            return __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_augadd(self):
        code = """
        def f(x, y):
            __start_extract__
            x += y
            __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            x = __f0(x, y)
            return x
        """
        post_extracted = """
        def __f0(__0, __1):
            __0 += __1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_augadd_const(self):
        code = """
        def f(x, y):
            __start_extract__
            x += 1
            __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            x = __f0(x)
            return x
        """
        post_extracted = """
        def __f0(__0):
            __0 += 1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_augadd_inside_function(self):
        code = """
        def f(x, y):
            def g(x):
                __start_extract__
                x += 1
                __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            def g(x):
                return __f0(x)
            return x
        """
        post_extracted = """
        def __f0(__0):
            __0 += 1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_do_not_add_return(self):
        code = """
        def f(x, y):
            __start_extract__
            x = y
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(y)
        """
        post_extracted = """
        def __f0(__1):
            __0 = __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_inside_if(self):
        code = """
        def f(x, y):
            if x > 0:
                __start_extract__
                x += y
                __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            if x > 0:
                x = __f0(x, y)
            return x
        """
        post_extracted = """
        def __f0(__0, __1):
            __0 += __1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_inside_else(self):
        code = """
        def f(x, y):
            if x > 0:
                x += y
            else:
                __start_extract__
                x += y
                __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x, y):
            if x > 0:
                x += y
            else:
                x = __f0(x, y)
            return x
        """
        post_extracted = """
        def __f0(__0, __1):
            __0 += __1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_conditional_early_return(self):
        code = """
        def f(x, y):
            __start_extract__
            if x > 0:
                return x
            x += 1
            __end_extract__
            x += y
            return x
        """
        self.assertEqual(self.run_extract(code), MultipleExits())

    def test_conditional_break(self):
        code = """
        def f(x, y):
            for _ in range(10):
                __start_extract__
                if x > 0:
                    break
                x += 1
                __end_extract__
            x += y
            return x
        """
        self.assertEqual(self.run_extract(code), MultipleExits())

    def test_undefined_input(self):
        code = """
        def f(x):
            if x > 0:
                y = 2
            __start_extract__
            if x > 3:
                x += y
            __end_extract__
            return x
        """
        self.assertEqual(self.run_extract(code), NonInitializedInputsOrOutputs())

    def test_undefined_output(self):
        code = """
        def f(x):
            __start_extract__
            if x > 3:
                y = 7
            __end_extract__
            if x > 5:
                x += y
            return x
        """
        self.assertEqual(self.run_extract(code), NonInitializedInputsOrOutputs())

    def test_nonlocal(self):
        code = """
        def f(x):
            def g():
                nonlocal x
                x = x + 1
                return x
            __start_extract__
            y = g()
            __end_extract__
            return y
        """
        self.assertEqual(
            self.run_extract(code),
            BannedComponentError("nonlocal", "us"),
        )

    def test_within_try(self):
        code = """
        def f(x):
            try:
                __start_extract__
                x = h(x)
                x = g(x)
                __end_extract__
            except:
                pass
            return x
        """
        self.assertEqual(self.run_extract(code), MultipleExits())

    def test_within_try_with_raise(self):
        code = """
        def f(x):
            try:
                __start_extract__
                raise RuntimeError
                __end_extract__
            except:
                y = 2
            return x
        """
        post_extract_expected = """
        def f(x):
            try:
                __f0()
            except:
                y = 2
            return x
        """
        post_extracted = """
        def __f0():
            raise RuntimeError
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_always_breaks(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                x = x + 1
                break
                __end_extract__
                x = x + 2
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
                break
                x = x + 2
            return x
        """
        post_extracted = """
        def __f0(__0):
            __0 = __0 + 1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_always_continues(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                x = x + 1
                continue
                __end_extract__
                x = x + 2
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
                continue
                x = x + 2
            return x
        """
        post_extracted = """
        def __f0(__0):
            __0 = __0 + 1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_always_breaks_in_if(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                if x > 0:
                    break
                else:
                    x = x + 1
                    break
                __end_extract__
                x = x + 2
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
                break
                x = x + 2
            return x
        """
        post_extracted = """
        def __f0(__0):
            if __0 > 0:
                return __0
            else:
                __0 = __0 + 1
                return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_always_continues_in_if(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                if x > 0:
                    continue
                else:
                    x = x + 1
                    continue
                __end_extract__
                x = x + 2
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
                continue
                x = x + 2
            return x
        """
        post_extracted = """
        def __f0(__0):
            if __0 > 0:
                return __0
            else:
                __0 = __0 + 1
                return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_normal_control_flow_if_autocontinue_anyway(self):
        code = """
        def f(x):
            while True:
                __start_extract__
                if x > 0:
                    continue
                else:
                    x = x + 1
                    continue
                __end_extract__
            return x
        """
        post_extract_expected = """
        def f(x):
            while True:
                x = __f0(x)
            return x
        """
        post_extracted = """
        def __f0(__0):
            if __0 > 0:
                return __0
            else:
                __0 = __0 + 1
                return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_y_then_x(self):
        code = """
        def f(x, y):
            __start_extract__
            y = y + 1
            x = x + 1
            __end_extract__
            return x, y
        """
        post_extract_expected = """
        def f(x, y):
            y, x = __f0(y, x)
            return x, y
        """
        post_extracted = """
        def __f0(__0, __1):
            __0 = __0 + 1
            __1 = __1 + 1
            return __0, __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_lambda_not_redefined(self):
        code = """
        def f(x, y):
            __start_extract__
            x = lambda: y
            __end_extract__
            return x, y
        """
        post_extract_expected = """
        def f(x, y):
            x = __f0(y)
            return x, y
        """
        post_extracted = """
        def __f0(__1):
            __0 = lambda: __1
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_lambda_redefined_internally(self):
        code = """
        def f(x, y):
            __start_extract__
            x = lambda: y
            y = 2
            __end_extract__
            return x
        """
        self.assertEqual(
            self.run_extract(code), ClosureOverVariableModifiedInExtractedCode()
        )

    def test_extract_lambda_redefined_externally(self):
        code = """
        def f(x, y):
            __start_extract__
            x = lambda: y
            __end_extract__
            y = 2
            return x
        """
        self.assertEqual(self.run_extract(code), ClosedVariablePassedDirectly())

    def test_extract_with_lambda(self):
        code = """
        def f(x, y):
            __start_extract__
            y = y + 1
            x = lambda: y
            __end_extract__
            return x, y
        """
        post_extract_expected = """
        def f(x, y):
            y, x = __f0(y)
            return x, y
        """
        post_extracted = """
        def __f0(__0):
            __0 = __0 + 1
            __1 = lambda: __0
            return __0, __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_not_in_scope_canonicalized_separately(self):
        code = """
        def f(y):
            __start_extract__
            x = sum(y)
            lambda z: z
            lambda y: y
            __end_extract__
        """
        post_extract_expected = """
        def f(y):
            return __f0(y)
        """
        post_extracted = """
        def __f0(__1):
            __0 = sum(__1)
            lambda __2: __2
            lambda __3: __3
        """
        self.assertCodes(
            self.run_extract(code, config=ExtractConfiguration(False)),
            (post_extract_expected, post_extracted),
        )

    def test_extract_with_lambda_reassign(self):
        code = """
        def f(x, y):
            __start_extract__
            y = y + 1
            x = lambda: y
            __end_extract__
            y = 2
            return x, y
        """
        self.assertEqual(
            self.run_extract(code), ClosureOverVariableModifiedInExtractedCode()
        )

    def test_extract_with_called_lambda(self):
        code = """
        def f(x, y):
            __start_extract__
            x = x ** 3
            z = lambda k: y ** (x - y) * (lambda x: x)(y) + k
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, y)
        """
        post_extracted = """
        def __f0(__0, __2):
            __0 = __0 ** 3
            __1 = lambda __3: __2 ** (__0 - __2) * (lambda __4: __4)(__2) + __3
        """
        self.assertCodes(
            self.run_extract(code, config=ExtractConfiguration(False)),
            (post_extract_expected, post_extracted),
        )

    def test_conditionally_initializing_a_variable_you_must_return(self):
        code = """
        def _main(i):
            count = 2
            __start_extract__
            if i == '8':
                count = 7
            __end_extract__
            return count
        """
        post_extract_expected = """
        def _main(i):
            count = 2
            count = __f0(i, count)
            return count
        """
        post_extracted = """
        def __f0(__0, __1):
            if __0 == '8':
                __1 = 7
            return __1
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_capturing_from_parent(self):
        code = """
        def f():
            func = 2
            def g():
                __start_extract__
                func()
                __end_extract__
            return g
        """
        post_extract_expected = """
        def f():
            func = 2
            def g():
                return __f0(func)
            return g
        """
        post_extracted = """
        def __f0(__0):
            __0()
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_site_defines_variable_referenced_in_closure_above(self):
        code = """
        def f():
            def g():
                return x
            __start_extract__
            x = 2
            __end_extract__
            return g
        """
        self.assertEqual(
            self.run_extract(code), ModifiesVariableClosedOverInNonExtractedCode()
        )

    def test_exception_multiple_exits_INCORRECT_TODO(self):
        code = """
        def test_3118(self):

            def gen():
                pass

            def f():
                g = gen()
                next(g)
                try:
                    __start_extract__
                    2
                    try:
                        raise {__metavariable__, __m0, ValueError}
                    except:
                        raise KeyError
                    __end_extract__
                except Exception as e:
                    self.assertIsInstance(e.__context__, ValueError)
            f()
        """
        # TODO this is not the correct output
        # It results from a bug in python_graphs that we should fix
        self.assertEqual(self.run_extract(code), MultipleExits())

    def test_imports(self):
        code = """
        def f():
            __start_extract__
            import os
            from collections import defaultdict
            import numpy as np
            import torch.nn as nn
            __end_extract__
            return os, defaultdict, np, nn
        """
        post_extract_expected = """
        def f():
            os, defaultdict, np, nn = __f0()
            return os, defaultdict, np, nn
        """
        post_extracted = """
        def __f0():
            import os as __0
            from collections import defaultdict as __1
            import numpy as __2
            import torch.nn as __3
            return __0, __1, __2, __3
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_global_TODO(self):
        # TODO we sohuld be able to handle this
        code = """
        def f():
            global x
            __start_extract__
            x = 2
            __end_extract__
            return x
        """
        post_extract_expected = """
        def f():
            global x
            __f0()
            return x
        """
        post_extracted = """
        def __f0():
            global x
            x = 2
        """
        self.assertEqual(self.run_extract(code), BannedComponentError("global", "us"))
        if not "fixed global":
            self.assertCodes(
                self.run_extract(code), (post_extract_expected, post_extracted)
            )

    def test_keyword_internally_safe_mode(self):
        code = """
        def f():
            __start_extract__
            def g(x):
                y = x + 2
                return y
            return g(x=2)
            __end_extract__
        """
        post_extract_expected = """
        def f():
            return __f0()
        """
        post_extracted = """
        def __f0():
            def __0(x):
                __1 = x + 2
                return __1
            return __0(x=2)
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_keyword_internally_unsafe_mode(self):
        code = """
        def f():
            __start_extract__
            def g(x):
                y = x + 2
                return y
            return g(x=2)
            __end_extract__
        """
        post_extract_expected = """
        def f():
            return __f0()
        """
        post_extracted = """
        def __f0():
            def __0(__1):
                __2 = __1 + 2
                return __2
            return __0(x=2)
        """
        self.assertCodes(
            self.run_extract(code, config=ExtractConfiguration(False)),
            (post_extract_expected, post_extracted),
        )

    def test_extract_with_del(self):
        code = """
        def f(x):
            __start_extract__
            y = x ** 2
            __end_extract__
            del y
        """
        post_extract_expected = """
        def f(x):
            y = __f0(x)
            del y
        """
        post_extracted = """
        def __f0(__1):
            __0 = __1 ** 2
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_generator(self):
        code = """
        def f():
            yield 1
            __start_extract__
            yield 2
            yield 3
            __end_extract__
        """
        post_extract_expected = """
        def f():
            yield 1
            yield from __f0()
            return
        """
        post_extracted = """
        def __f0():
            yield 2
            yield 3
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_extract_generator_also_returns(self):
        code = """
        def f():
            yield 1
            __start_extract__
            yield 2
            yield 3
            x = 7
            __end_extract__
            print(x)
        """
        self.assertEqual(self.run_extract(code), BothYieldsAndReturns())

    def test_coroutine_banned(self):
        code = """
        def f():
            __start_extract__
            return [(yield 2)]
            __end_extract__
        """
        self.assertEqual(
            self.run_extract(code), BannedComponentError("coroutine", "us")
        )

    def test_default_argument_in_child_function(self):
        code = """
        def f(x):
            __start_extract__
            y = {__metavariable__, __m0, lambda i=x: i}
            __end_extract__
            return y
        """
        post_extract_expected = """
        def f(x):
            y = __f0(lambda: lambda i=x: i)
            return y
        """
        post_extracted = """
        def __f0(__m0):
            __0 = __m0()
            return __0
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_affects_same_variable(self):
        code = """
        def g(x):
            __start_extract__
            x = {__metavariable__, __m0, [x]}
            __end_extract__
        """
        post_extract_expected = """
        def g(x):
            return __f0(lambda: [x])
        """
        post_extracted = """
        def __f0(__m0):
            __0 = __m0()
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_temp(self):
        code = """
        def f():
            for (video, subtitles) in subtitles_by_video.iteritems():
                subtitles = 2
                __start_extract__
                {__metavariable__, __m0, subtitles.sort}
                __end_extract__
        """

        post_extract_expected = """
        def f():
            for (video, subtitles) in subtitles_by_video.iteritems():
                subtitles = 2
                __f0(lambda: subtitles.sort(key=lambda s: key_subtitles(s, video, languages, services, order), reverse=True))
        """
        post_extracted = """
        def __f0(__m0):
            __m0()
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )


class GenericExtractRealisticTest(GenericExtractTest):
    def test_temporary(self):
        try:
            with open("__test__.py") as f:
                code = f.read()
        except FileNotFoundError:
            return
        self.run_extract(code)

    def operate(self, i, use_full_tree=False):
        code = small_set_examples()[i]
        return self.operate_on_code(i, code, use_full_tree=use_full_tree)

    def operate_on_code(self, seed, full_code, use_full_tree=False):
        rng = np.random.RandomState(seed)
        tree = ast.parse(full_code)
        g = control_flow.get_control_flow_graph(tree)
        num_entry_points = len(list(g.get_enter_blocks()))
        print(full_code)
        result = []
        for i in range(num_entry_points):
            tree = ast.parse(full_code)
            g = control_flow.get_control_flow_graph(tree)
            entry_point = list(g.get_enter_blocks())[i]
            print(entry_point)
            if not entry_point.node.body:
                continue
            code, count = self.sample_site(rng, entry_point.node)
            if use_full_tree:
                code = ast.unparse(tree)
            print(code)
            try:
                out = self.run_extract(code, count)
                result.append((code, out))
            except (NotApplicable, BannedComponentError):
                # don't error on this, just skip it
                pass
            except SyntaxError:
                self.fail()
        return result

    def sample_site(self, rng, tree):
        nodes = list(ast_nodes_in_order(tree))
        fields = [
            (n, f)
            for n in nodes
            for f in n._fields
            if ns.python_ast_tools.field_is_body(type(n), f) and len(getattr(n, f)) > 0
        ]
        node, field = fields[rng.choice(len(fields))]
        length = len(getattr(node, field))
        start = rng.randint(length)
        end = rng.randint(start + 1, length + 1)
        body = getattr(node, field)
        body.insert(start, ast.Expr(ast.Name("__start_extract__")))
        body.insert(end + 1, ast.Expr(ast.Name("__end_extract__")))
        body[start + 1 : end + 1], count = self.manipulate(
            body[start + 1 : end + 1], rng
        )
        return ast.unparse(tree), count

    def manipulate(self, body, rng):
        del rng
        return body, 0


class ExtractRealisticTest(GenericExtractRealisticTest):
    @expand_with_slow_tests(len(small_set_examples()))
    def test_realistic(self, i):
        self.operate(i)
