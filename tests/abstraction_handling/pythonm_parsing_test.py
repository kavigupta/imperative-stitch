import ast
import json
from textwrap import dedent
import unittest

import neurosym as ns

from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies_recursively,
)
from imperative_stitch.compress.pythonm import (
    parse_pythonm,
    parse_with_target_state,
    replace_pythonm_with_normal_stub,
)
from imperative_stitch.compress.rust_stitch.process_rust_stitch import (
    handle_splice_seqs,
)
from tests.utils import canonicalize, expand_with_slow_tests, small_set_examples

from .abstraction_test import assertSameCode, fn_2, fn_2_args_w_nothing


def fix_symbols(s_expr):
    if isinstance(s_expr, ns.SExpression):
        return ns.SExpression(
            s_expr.symbol,
            [fix_symbols(child) for child in s_expr.children],
        )
    assert isinstance(s_expr, str), s_expr
    if s_expr.startswith("&"):
        return s_expr[1:].split(":")[0]
    if s_expr.startswith("g_"):
        return s_expr[2:].split(":")[0]
    return s_expr


def assertSameSExprUpToSymbolScopes(
    testcase, original_code, rewritten_pythonm, dfa_root
):
    """
    Assert that the original code and the rewritten PythonM code are the same up to symbol scopes.
    This is useful for testing that the PythonM parsing works correctly.
    """
    original_code = parse_with_target_state(original_code, dfa_root)
    testcase.maxDiff = None

    original_s_expr = ns.render_s_expression(
        handle_splice_seqs(fix_symbols(original_code.to_ns_s_exp()))
    )
    rewritten_s_expr = ns.render_s_expression(
        handle_splice_seqs(fix_symbols(rewritten_pythonm.to_ns_s_exp()))
    )

    testcase.assertEqual(original_s_expr, rewritten_s_expr)


def assertPythonMParsingWorks(
    testcase, original_code, rewritten_pythonm, abstractions, dfa_root="M"
):
    stub_back_forth = parse_pythonm(rewritten_pythonm, dfa_root, abstractions)

    back_forth_inlined = abstraction_calls_to_bodies_recursively(
        stub_back_forth, abstractions
    )

    assertSameSExprUpToSymbolScopes(
        testcase, original_code, back_forth_inlined, dfa_root
    )

    testcase.assertEqual(ast.unparse(ast.parse(original_code)), back_forth_inlined.to_python())


class TestPythonMParsing(unittest.TestCase):

    def test_stub_parsing(self):
        abstractions = {"fn_2": fn_2}
        args = fn_2_args_w_nothing[:-1] + [
            ns.python_statements_to_python_ast("x = 2 if 3 else 9")
        ]
        stub = fn_2.create_stub(args, is_pythonm=True).to_python()
        original_inlined = fn_2.substitute_body(args).to_python()
        assertSameCode(
            self,
            stub,
            """
            fn_2(`print(2)`, &c, &a, &b, &d, `x = 2 if 3 else 9`)
            """,
        )

        assertPythonMParsingWorks(
            self, original_inlined, stub, abstractions, fn_2.dfa_root
        )

    def test_stub_parsing_last_token_long(self):
        abstractions = {"fn_2": fn_2}
        args = fn_2_args_w_nothing[:-1] + [
            ns.python_statements_to_python_ast("x = 2 if 3 else 929473294823")
        ]
        stub = fn_2.create_stub(args, is_pythonm=True).to_python()
        original_inlined = fn_2.substitute_body(args).to_python()
        assertSameCode(
            self,
            stub,
            """
            fn_2(`print(2)`, &c, &a, &b, &d, `x = 2 if 3 else 929473294823`)
            """,
        )

        assertPythonMParsingWorks(
            self, original_inlined, stub, abstractions, fn_2.dfa_root
        )

    def test_stub_parsing_w_empty(self):
        abstractions = {"fn_2": fn_2}
        args = fn_2_args_w_nothing
        stub = fn_2.create_stub(args, is_pythonm=True).to_python()
        original_inlined = fn_2.substitute_body(args).to_python()
        assertSameCode(
            self,
            stub,
            """
            fn_2(`print(2)`, &c, &a, &b, &d, ``)
            """,
        )

        self.assertEqual(
            replace_pythonm_with_normal_stub(stub),
            "fn_2(__code__('print(2)'), __ref__(c), __ref__(a), __ref__(b), __ref__(d), __code__(''))",
        )

        assertPythonMParsingWorks(
            self, original_inlined, stub, abstractions, fn_2.dfa_root
        )

    def assertParseBasicNoAbstrs(self, code):
        """
        Assert that the code can be parsed as PythonM and that it does not contain any abstractions.
        """
        rewritten_pythonm = parse_pythonm(code, "M", {})
        assertSameCode(self, canonicalize(code), rewritten_pythonm.to_python())

    def test_parse_basic_not_containing_abstrs(self):
        self.assertParseBasicNoAbstrs('x = "`abc`"')
        self.assertParseBasicNoAbstrs("x = 2 & 3")
        self.assertParseBasicNoAbstrs("x = '2 & 3'")

    @expand_with_slow_tests(1000, first_fast=5)
    def test_smoke_small_set(self, seed):
        programs = small_set_examples()[seed]
        self.assertParseBasicNoAbstrs(programs)

    @expand_with_slow_tests(1000, first_fast=5)
    def test_smoke_vlm_human(self, seed):
        with open("data/vlmaterial-set/human_1000.json", "r") as f:
            programs = json.load(f)
        self.assertParseBasicNoAbstrs(programs[seed])

    def test_multiple_empty_codeblocks(self):
        self.assertEqual(
            replace_pythonm_with_normal_stub(
                "fn_0(&name, &new, ``) + fn_0(&name, &new, ``)"
            ),
            "fn_0(__ref__(name), __ref__(new), __code__('')) + fn_0(__ref__(name), __ref__(new), __code__(''))",
        )

    def test_multiline_strings_with_trailing_spaces(self):
        code = dedent(
            '''
            """
                abc
                x
                def
            """
            '''
        ).replace("x", "")

        assertPythonMParsingWorks(self, code, code, {})
