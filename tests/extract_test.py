import ast
from textwrap import dedent
import unittest
from imperative_stitch.analyze_program.extract.errors import (
    MultipleExits,
    NonInitializedInputs,
    NonInitializedOutputs,
)
from imperative_stitch.analyze_program.ssa.banned_component import BannedComponentError

from imperative_stitch.data import parse_extract_pragma
from imperative_stitch.analyze_program.extract import (
    do_extract,
    replace_break_and_continue,
    remove_unnecessary_returns,
)
from imperative_stitch.analyze_program.extract import NotApplicable


def canonicalize(code):
    code = dedent(code)
    code = ast.unparse(ast.parse(code))
    return code


class ReplaceBreakAndContinueTest(unittest.TestCase):
    def run_replace(self, code):
        code = canonicalize(code)
        tree = ast.parse(code)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        tree = tree.body[0]
        b, c, tree = replace_break_and_continue(
            tree, ast.Expr(ast.Name("__replaced__"))
        )
        return b, c, ast.unparse(tree)

    def assertCodes(self, pre_replace, post_replace, b=False, c=False):
        post_replace = canonicalize(post_replace)
        b_ac, c_ac, post_replace_ac = self.run_replace(pre_replace)
        self.assertEqual(b, b_ac)
        self.assertEqual(c, c_ac)
        self.assertEqual(post_replace, post_replace_ac)

    def test_basic(self):
        pre_replace = """
        def f(x):
            break
            continue
            return x
        """
        post_replace = """
        def f(x):
            __replaced__
            __replaced__
            return x
        """
        self.assertCodes(pre_replace, post_replace, b=True, c=True)

    def test_in_loop(self):
        pre_replace = """
        def f(x):
            while True:
                break
                continue
                return x
        """
        post_replace = """
        def f(x):
            while True:
                break
                continue
                return x
        """
        self.assertCodes(pre_replace, post_replace)

    def test_inner_function(self):
        pre_replace = """
        def f(x):
            def g():
                break
                continue
                return x
            return g()
        """
        post_replace = """
        def f(x):
            def g():
                break
                continue
                return x
            return g()
        """
        self.assertCodes(pre_replace, post_replace)

    def test_both_in_loop_and_out(self):
        pre_replace = """
        def f(x):
            break
            while True:
                continue
                return x
        """
        post_replace = """
        def f(x):
            __replaced__
            while True:
                continue
                return x
        """
        self.assertCodes(pre_replace, post_replace, b=True)


class RemoveUnnecessaryReturnsTest(unittest.TestCase):
    def run_remove(self, code):
        code = canonicalize(code)
        tree = ast.parse(code)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        tree = tree.body[0]
        tree = remove_unnecessary_returns(tree)
        return ast.unparse(tree)

    def assertCode(self, pre_remove, post_remove):
        post_remove = canonicalize(post_remove)
        post_replace_ac = self.run_remove(pre_remove)
        self.assertEqual(post_remove, post_replace_ac)

    def test_then_empty(self):
        pre_remove = """
        def f(x):
            return
        """
        post_remove = """
        def f(x):
            pass
        """
        self.assertCode(pre_remove, post_remove)

    def test_non_empty(self):
        pre_remove = """
        def f(x):
            x = 2
            return
        """
        post_remove = """
        def f(x):
            x = 2
        """
        self.assertCode(pre_remove, post_remove)

    def test_necessary_return_at_end(self):
        pre_remove = """
        def f(x):
            return x
        """
        post_remove = """
        def f(x):
            return x
        """
        self.assertCode(pre_remove, post_remove)

    def test_blank_return_not_at_end(self):
        pre_remove = """
        def f(x):
            return
            x = 2
        """
        post_remove = """
        def f(x):
            return
            x = 2
        """
        self.assertCode(pre_remove, post_remove)

    def test_dead_return_sequence(self):
        pre_remove = """
        def f(x):
            return 2
            return x
        """
        post_remove = """
        def f(x):
            return 2
        """
        self.assertCode(pre_remove, post_remove)

    def test_dead_return_sequence_post_if(self):
        pre_remove = """
        def f(x):
            if x > 0:
                return 2
            else:
                return x
            return 3
        """
        post_remove = """
        def f(x):
            if x > 0:
                return 2
            else:
                return x
        """
        self.assertCode(pre_remove, post_remove)


class ExtractTest(unittest.TestCase):
    def run_extract(self, code):
        code = canonicalize(code)
        tree, [site] = parse_extract_pragma(code)
        # without pragmas
        code = ast.unparse(tree)
        try:
            func_def, undo = do_extract(site, tree, extract_name="__f0")
        except (NotApplicable, BannedComponentError) as e:
            # ensure that the code is not changed
            self.assertEqual(code, ast.unparse(tree))
            return e

        post_extract, extracted = ast.unparse(tree), ast.unparse(func_def)
        undo()
        self.assertEqual(code, ast.unparse(tree), "undo")
        return post_extract, extracted

    def assertCodes(self, expected, actual):
        post_extract, extracted = actual
        expected_post_extract, expected_extracted = expected
        self.assertEqual(
            canonicalize(expected_post_extract),
            canonicalize(post_extract),
            "post_extract",
        )
        self.assertEqual(
            canonicalize(expected_extracted), canonicalize(extracted), "extracted"
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
        def __f0(x, y):
            x, y = y, x
            z = x + y
            if z > 0:
                x += 1
            return x, y
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
        def __f0(x, y):
            x2y2 = x ** 2 + y ** 2
            r = x2y2 ** 0.5
            z = x + r
            return z
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
        def __f0(x):
            x = x + 1
            return x
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
        def __f0(x, y):
            if x > 0:
                return x
            else:
                return y
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
        def __f0(x, y):
            if x > 0:
                return x
            y += 1
            return y
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
        def __f0(x, y):
            x += y
            return x
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
        def __f0(y):
            x = y
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
        self.assertEqual(self.run_extract(code), NonInitializedInputs())

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
        self.assertEqual(self.run_extract(code), NonInitializedOutputs())

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
            BannedComponentError(
                "nonlocal statements cannot be used because we do not support them yet"
            ),
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
        post_extract_expected = """
        def f(x):
            try:
                x = __f0(x)
            except:
                pass
            return x
        """
        post_extracted = """
        def __f0(x):
            x = h(x)
            x = g(x)
            return x
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
        def __f0(x):
            x = x + 1
            return x
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
        def __f0(x):
            x = x + 1
            return x
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
        def __f0(x):
            if x > 0:
                return x
            else:
                x = x + 1
                return x
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
        def __f0(x):
            if x > 0:
                return x
            else:
                x = x + 1
                return x
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
        def __f0(x):
            if x > 0:
                return x
            else:
                x = x + 1
                return x
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )