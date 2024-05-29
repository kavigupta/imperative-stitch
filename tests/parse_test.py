import ast
import copy
import unittest

import neurosym as ns
from increase_recursionlimit import increase_recursionlimit
from parameterized import parameterized

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    collect_abstraction_calls,
    replace_abstraction_calls,
)
from imperative_stitch.data.stitch_output_set import (
    load_stitch_output_set,
    load_stitch_output_set_no_dfa,
)
from imperative_stitch.parser import converter
from imperative_stitch.utils.classify_nodes import export_dfa
from tests.abstraction_handling.abstraction_test import assertSameCode
from tests.utils import expand_with_slow_tests, small_set_examples


class ParseUnparseInverseTest(unittest.TestCase):
    def canonicalize(self, python_code):
        return ast.unparse(ast.parse(python_code))

    def assert_valid_s_exp(self, s_exp, no_leaves):
        if not isinstance(s_exp, ns.SExpression):
            assert isinstance(s_exp, str)
            if no_leaves:
                self.fail(f"leaf: {s_exp}")
            return
        if s_exp.symbol not in {"/seq"}:
            self.assertTrue(isinstance(s_exp.symbol, str), repr(s_exp.symbol))
            if not no_leaves:
                self.assertTrue(len(s_exp.children) >= 1, repr(s_exp))
        for y in s_exp.children:
            self.assert_valid_s_exp(y, no_leaves)

    def check_s_exp(self, s_exp, no_leaves=False):
        self.maxDiff = None
        parsed = converter.s_exp_to_python_ast(s_exp)
        print(parsed)
        with increase_recursionlimit():
            s_exp_update = ns.render_s_expression(ns.parse_s_expression(s_exp))
        self.assertEqual(
            s_exp_update,
            ns.render_s_expression(parsed.to_ns_s_exp(dict(no_leaves=no_leaves))),
        )

    def check_with_args(self, test_code, no_leaves=False):
        test_code = self.canonicalize(test_code)
        s_exp = converter.python_to_s_exp(
            test_code, renderer_kwargs=dict(columns=80), no_leaves=no_leaves
        )
        with increase_recursionlimit():
            self.assert_valid_s_exp(ns.parse_s_expression(s_exp), no_leaves=no_leaves)
        self.check_s_exp(s_exp, no_leaves=no_leaves)
        s_exp_parsed = converter.s_exp_to_python_ast(s_exp)
        print(repr(s_exp_parsed))
        modified = converter.s_exp_to_python(s_exp)
        self.assertEqual(test_code, modified)

    def check(self, test_code):
        self.check_with_args(test_code)
        self.check_with_args(test_code, no_leaves=True)

    def test_basic_one_liners(self):
        self.check("x = 2")
        self.check("7")
        self.check("import abc")

    def test_imports(self):
        self.check("import os")
        self.check("from os import path")
        self.check("import os.path")

    def test_sequence_of_statements(self):
        self.maxDiff = None
        self.check("x = 2\ny = 3\nz = 4")
        self.assertEqual(
            converter.python_to_s_exp(
                "x = 2\ny = 3\nz = 4", renderer_kwargs=dict(columns=80000)
            ),
            "(Module (/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None) (Assign (list (Name &y:0 Store)) (Constant i3 None) None) (Assign (list (Name &z:0 Store)) (Constant i4 None) None)) nil)",
        )

    def test_globals(self):
        self.assertEqual(
            converter.python_to_s_exp("import os", renderer_kwargs=dict(columns=80)),
            "(Module (/seq (Import (list (alias g_os None)))) nil)",
        )

    def test_builtins(self):
        self.check("print(True)")
        self.check("0")
        self.check("x = None")

    def test_if_expr(self):
        self.check("2 if x == 3 else 4")

    def test_strings(self):
        self.check("' '")
        self.check("x = 'abc '")
        self.check("x = 'a=é b=\\udc80 d=\U0010ffff'")
        self.check("x = '\\uABCD'")

    def test_lambda(self):
        self.check("lambda: 1 + 2")

    def test_subscript_basic(self):
        self.check("x[2]")

    def test_subscript_tuple(self):
        self.check("x[2, 3]")

    def test_if(self):
        self.check("if True: pass")

    def test_unparse_sequence(self):
        # should work with or without the Module wrapper
        self.assertEqual(
            converter.s_exp_to_python(
                "(Module (/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None)) nil)"
            ),
            "x = 2",
        )

        self.assertEqual(
            converter.s_exp_to_python(
                "(/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None))"
            ),
            "x = 2",
        )

    @expand_with_slow_tests(len(small_set_examples()), 100)
    def test_realistic(self, i):
        try:
            print(small_set_examples()[i])
            self.check(small_set_examples()[i])
        except Exception as e:
            self.assertFalse(f"Error: {e}")
            raise e

    def test_subscript_s_exp(self):
        self.check_s_exp(
            "(_slice_tuple (Tuple (list (_slice_content (Constant i3 None))) Load))"
        )


class AbstractionBodyRenderTest(unittest.TestCase):
    basic_symvars = "(Assign (list (Name %1 Store)) (Constant i2 None) None)"

    def test_basic_symvars_variables_python(self):
        body = converter.s_exp_to_python_ast(self.basic_symvars)
        assertSameCode(
            self,
            "%1 = 2",
            body.to_python(),
        )

    def test_basic_symvars_variables_s_exp(self):
        self.maxDiff = None
        body = converter.s_exp_to_python_ast(self.basic_symvars)
        self.assertEqual(
            self.basic_symvars,
            ns.render_s_expression(body.to_ns_s_exp()),
        )
        self.assertEqual(
            "(Assign (list (Name (var-%1) (Store))) (Constant (const-i2) (const-None)) (const-None))",
            ns.render_s_expression(body.to_ns_s_exp(dict(no_leaves=True))),
        )

    def test_basic_symvars_variables_type_annotated_s_exp(self):
        body = converter.s_exp_to_python_ast(self.basic_symvars)
        self.assertEqual(
            ns.parse_s_expression(
                """
                (Assign~S
                    (list~_L_~1 (Name~L (var-%1~Name) (Store~Ctx)))
                    (Constant~E (const-i2~Const) (const-None~ConstKind)) (const-None~TC))
                """
            ),
            converter.python_ast_to_type_annotated_s_exp(body, export_dfa(), "S"),
        )

    def test_sequence_metavar_s_exp_export(self):
        self.maxDiff = None
        result = converter.python_ast_to_type_annotated_s_exp(
            converter.s_exp_to_python_ast(
                """
                (FunctionDef
                    &f:0
                    (arguments nil
                    (list (arg &x:1 None None))
                    None nil nil None nil)
                    #0 nil None None)
                """
            ),
            export_dfa(),
            "S",
        )
        result = ns.render_s_expression(result)
        expected = ns.render_s_expression(
            ns.parse_s_expression(
                """
                (FunctionDef~S
                    (const-&f:0~Name)
                    (arguments~As
                        (list~_A_~0)
                        (list~_A_~1 (arg~A (const-&x:1~Name) (const-None~TA) (const-None~TC)))
                        (const-None~A) (list~_A_~0) (list~_E_~0) (const-None~A) (list~_E_~0))
                    (var-#0~seqS) (list~_E_~0) (const-None~TA) (const-None~TC))
                """
            )
        )
        print(result)
        print(expected)
        self.assertEqual(expected, result)

    all_kinds = """
    (/seq
        (Assign
            (list (Name %1 Store))
            (BinOp (Constant i2 None) Add #0))
        ?0
        (While
            (Constant True None)
            (/seq (Expr
                (Compare
                    (Name %1 Load)
                    (list Eq)
                    (list #1))))
            (/seq))
        ?1
    )
    """

    def test_all_kinds_python(self):
        body = converter.s_exp_to_python_ast(self.all_kinds)
        # not amazing rendering but it's fine
        assertSameCode(
            self,
            """
            %1 = 2 + #0?0
            while True:
                %1 == #1?1
            """,
            body.to_python(),
        )

    def test_all_kinds_s_exp(self):
        body = converter.s_exp_to_python_ast(self.all_kinds)
        self.assertEqual(
            converter.s_exp_to_python_ast(self.all_kinds).to_ns_s_exp(),
            body.to_ns_s_exp(),
        )

    def test_all_kinds_type_annotated_s_exp(self):
        self.maxDiff = None
        body = converter.s_exp_to_python_ast(self.all_kinds)
        self.assertEqual(
            ns.render_s_expression(
                ns.parse_s_expression(
                    """
                    (/seq~seqS~4
                        (Assign~S
                            (list~_L_~1 (Name~L (var-%1~Name) (Store~Ctx)))
                            (BinOp~E
                                (Constant~E (const-i2~Const) (const-None~ConstKind))
                                (Add~O)
                                (var-#0~E)))
                        (var-?0~S)
                        (While~S
                            (Constant~E (const-True~Const) (const-None~ConstKind))
                            (/seq~seqS~1
                                (Expr~S
                                    (Compare~E
                                        (Name~E (var-%1~Name) (Load~Ctx))
                                        (list~_O_~1 (Eq~O))
                                        (list~_E_~1 (var-#1~E))
                                    )
                                )
                            )
                            (/seq~seqS~0)
                        )
                        (var-?1~S)
                    )
                    """
                )
            ),
            ns.render_s_expression(
                converter.python_ast_to_type_annotated_s_exp(body, export_dfa(), "seqS")
            ),
        )


class AbstractionCallsTest(unittest.TestCase):
    ctx_in_seq = """
    (Module
        (/seq
            (/splice
                (fn_1 &n:0 &s:0))
            (Assign (list (Name &k:0 Store)) (Call (Attribute (Name &s:0 Load) s_count Load) (list (Constant s_8 None)) nil) None))
        nil)
    """

    ctx_rooted = """
    (Module
        (/seq
            (If
                (Name g_x Load)
                (fn_1 &a:0 &z:0)
                nil))
        nil)
    """

    def test_gather_calls(self):
        calls = collect_abstraction_calls(
            converter.s_exp_to_python_ast(self.ctx_in_seq)
        )
        self.assertEqual(len(calls), 1)
        abstraction_calls = [
            ns.render_s_expression(x.to_ns_s_exp()) for x in calls.values()
        ]
        self.assertEqual(sorted(abstraction_calls), ["(fn_1 &n:0 &s:0)"])

    def test_substitute_in_seq(self):
        seq = converter.s_exp_to_python_ast(self.ctx_in_seq)
        out = {
            x: converter.python_statements_to_python_ast("x = 2; x = 3")
            for x in collect_abstraction_calls(seq)
        }
        substituted = replace_abstraction_calls(seq, out)

        assertSameCode(
            self,
            """
            x = 2
            x = 3
            k = s.count('8')
            """,
            substituted.to_python(),
        )

    def test_substitute_in_rooted(self):
        seq = converter.s_exp_to_python_ast(self.ctx_rooted)
        out = {
            x: converter.python_statements_to_python_ast("x = 2; x = 3")
            for x in collect_abstraction_calls(seq)
        }
        substituted = replace_abstraction_calls(seq, out)

        assertSameCode(
            self,
            """
            if x:
                x = 2
                x = 3
            """,
            substituted.to_python(),
        )

    def assertParseUnparseSExp(self, program):
        program = ns.render_s_expression(ns.parse_s_expression(program))
        program_procd = ns.render_s_expression(
            converter.s_exp_to_python_ast(program).to_ns_s_exp(dict(no_leaves=False))
        )
        self.assertEqual(program, program_procd)

    def test_starred_content_with_abstraction(self):
        self.assertParseUnparseSExp("(_starred_content (fn_23))")

    def test_choiceseq(self):
        self.assertParseUnparseSExp("(fn_3 (/choiceseq (fn_3 (/choiceseq))))")

    def test_choiceseq_abstr(self):
        self.assertParseUnparseSExp("(fn_3 (fn_3 (/choiceseq)))")

    def test_slice_content_with_abstraction(self):
        self.assertParseUnparseSExp("(_slice_content (fn_23))")

    def test_starred_content_with_abstraction_medium(self):
        self.assertParseUnparseSExp(
            """
            (Assign
                (list (Name &x:0 Store))
                (Tuple (list (_starred_content (Constant i2 None)) (_starred_content (fn_1))) Load)
                None
            )
            """
        )

    def test_starred_content_with_abstraction_large_test(self):
        self.assertParseUnparseSExp(
            """
            (Module
                (/seq
                    (FunctionDef
                        &find_path:0
                        (arguments
                            nil
                            (list
                                (arg &N:1 None None)
                                (arg &K:1 None None)
                                (arg &items:1 None None)
                                (arg &M:1 None None)
                                (arg &purchase:1 None None)
                            )
                            None
                            nil
                            nil
                            None
                            nil
                        )
                        (/seq
                            (/splice (fn_3 &item_to_store:1 &item:1 &i:1 &items:1 (/choiceseq)))
                            (Assign (list (Name &possible_paths:1 Store)) (Constant i1 None) None)
                            (Assign (list (Name &last_store:1 Store)) (UnaryOp USub (Constant i1 None)) None)
                            (For
                                (Name &item:1 Store)
                                (Name &purchase:1 Load)
                                (/seq
                                    (If
                                        (Compare
                                            (Name &item:1 Load)
                                            (list In)
                                            (list (Name &item_to_store:1 Load))
                                        )
                                        (/seq
                                            (Assign
                                                (list (Name &stores_with_item:1 Store))
                                                (Subscript
                                                    (Name &item_to_store:1 Load)
                                                    (_slice_content (Name &item:1 Load))
                                                    Load
                                                )
                                                None
                                            )
                                            (Assign
                                                (list (Name &stores_with_item:1 Store))
                                                (ListComp
                                                    (Name &store:2 Load)
                                                    (list
                                                        (comprehension
                                                            (Name &store:2 Store)
                                                            (Name &stores_with_item:1 Load)
                                                            (list
                                                                (Compare
                                                                    (Name &store:2 Load)
                                                                    (list Gt)
                                                                    (list (Name &last_store:1 Load))
                                                                )
                                                            )
                                                            i0
                                                        )
                                                    )
                                                )
                                                None
                                            )
                                            (If
                                                (UnaryOp Not (Name &stores_with_item:1 Load))
                                                (/seq (Return (Constant s_impossible None)))
                                                (/seq
                                                    (If
                                                        (fn_8 &stores_with_item:1)
                                                        (/seq
                                                            (AugAssign
                                                                (Name &possible_paths:1 Store)
                                                                Mult
                                                                (Call
                                                                    (Name g_len Load)
                                                                    (list
                                                                        (_starred_content
                                                                            (Name &stores_with_item:1 Load)
                                                                        )
                                                                    )
                                                                    nil
                                                                )
                                                            )
                                                            (Assign
                                                                (list (Name &last_store:1 Store))
                                                                (Call
                                                                    (Name g_min Load)
                                                                    (list
                                                                        (_starred_content
                                                                            (Name &stores_with_item:1 Load)
                                                                        )
                                                                    )
                                                                    nil
                                                                )
                                                                None
                                                            )
                                                        )
                                                        (/seq
                                                            (Assign
                                                                (list (Name &last_store:1 Store))
                                                                (Subscript
                                                                    (Name &stores_with_item:1 Load)
                                                                    (_slice_content (Constant i0 None))
                                                                    Load
                                                                )
                                                                None
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        (/seq (Return (Constant s_impossible None)))
                                    )
                                )
                                (/seq)
                                None
                            )
                            (fn_22 (Name &possible_paths:1 Load))
                        )
                        nil
                        None
                        None
                    )
                    (Assign
                        (list (Name &N:0 Store))
                        (Call (Name g_int Load) (list (_starred_content (fn_23))) nil)
                        None
                    )
                    (Assign
                        (list (Name &K:0 Store))
                        (Call (Name g_int Load) (list (_starred_content (fn_23))) nil)
                        None
                    )
                    (Assign (list (Name &items:0 Store)) (List nil Load) None)
                    (For
                        (Name &_:0 Store)
                        (Call (Name g_range Load) (list (_starred_content (Name &K:0 Load))) nil)
                        (/seq
                            (Assign
                                (list
                                    (Tuple
                                        (list
                                            (_starred_content (Name &i:0 Store))
                                            (_starred_content (Name &item:0 Store))
                                        )
                                        Store
                                    )
                                )
                                (fn_11)
                                None
                            )
                            (Expr
                                (Call
                                    (Attribute (Name &items:0 Load) s_append Load)
                                    (list
                                        (_starred_content
                                            (Tuple
                                                (list
                                                    (_starred_content
                                                        (Call
                                                            (Name g_int Load)
                                                            (list (_starred_content (Name &i:0 Load)))
                                                            nil
                                                        )
                                                    )
                                                    (_starred_content (Name &item:0 Load))
                                                )
                                                Load
                                            )
                                        )
                                    )
                                    nil
                                )
                            )
                        )
                        (/seq)
                        None
                    )
                    (Assign
                        (list (Name &M:0 Store))
                        (Call (Name g_int Load) (list (_starred_content (fn_23))) nil)
                        None
                    )
                    (Assign (list (Name &purchase:0 Store)) (List nil Load) None)
                    (For
                        (Name &_:0 Store)
                        (Call (Name g_range Load) (list (_starred_content (Name &M:0 Load))) nil)
                        (/seq
                            (Expr
                                (Call
                                    (Attribute (Name &purchase:0 Load) s_append Load)
                                    (list (_starred_content (fn_23)))
                                    nil
                                )
                            )
                        )
                        (/seq)
                        None
                    )
                    (Assign
                        (list (Name &result:0 Store))
                        (Call
                            (Name &find_path:0 Load)
                            (list
                                (_starred_content (Name &N:0 Load))
                                (_starred_content (Name &K:0 Load))
                                (_starred_content (Name &items:0 Load))
                                (_starred_content (Name &M:0 Load))
                                (_starred_content (Name &purchase:0 Load))
                            )
                            nil
                        )
                        None
                    )
                    (Expr (Call (Name g_print Load) (list (_starred_content (Name &result:0 Load))) nil))
                )
                nil
            )
            """
        )


class AbstractionBodiesTest(unittest.TestCase):
    def check_abstractions_in(self, x):
        x = copy.deepcopy(x)
        prev_abstrs = []
        for i, abstr in enumerate(x["abstractions"], 1):
            body = converter.s_exp_to_python_ast(abstr["body"])
            body_ns_s_exp = ns.render_s_expression(
                converter.python_ast_to_type_annotated_s_exp(
                    body, export_dfa(abstrs=prev_abstrs), abstr["dfa_root"]
                )
            )
            body_from_ns_s_exp = ns.render_s_expression(
                converter.s_exp_to_python_ast(body_ns_s_exp).to_ns_s_exp()
            )
            self.assertEqual(abstr["body"], body_from_ns_s_exp)
            prev_abstrs.append(Abstraction.of(f"fn_{i}", **abstr))

    @parameterized.expand(range(len(load_stitch_output_set())))
    def test_realistic_with_abstractions(self, i):
        self.check_abstractions_in(load_stitch_output_set()[i])

    @parameterized.expand(range(len(load_stitch_output_set_no_dfa())))
    def test_realistic_with_abstractions_no_dfa(self, i):
        self.check_abstractions_in(load_stitch_output_set_no_dfa()[i])
