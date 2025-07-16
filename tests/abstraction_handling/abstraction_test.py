import ast
import copy
import unittest
from textwrap import dedent
import ast

import neurosym as ns
from parameterized import parameterized

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies,
    abstraction_calls_to_bodies_recursively,
    collect_abstraction_calls,
    replace_abstraction_calls,
)
from imperative_stitch.data.stitch_output_set import (
    load_stitch_output_set,
    load_stitch_output_set_no_dfa,
)
from imperative_stitch.parser import converter
from imperative_stitch.utils.classify_nodes import export_dfa
from imperative_stitch.utils.def_use_mask_extension.ordering import (
    python_node_ordering_with_abstractions,
)
from tests.utils import (
    expand_with_slow_tests,
    load_annies_compressed_individual_programs,
)

fn_1_body = """
(/subseq
    (Assign
        (list (Name %1 Store))
        (Call (Name g_int Load) (list (_starred_content (Call (Name g_input Load) nil nil))) nil) None)
    (Assign (list (Name %2 Store)) (Call (Name g_input Load) nil nil) None))
"""

fn_1 = Abstraction.of("fn_1", fn_1_body, "seqS", dfa_symvars=["X", "X"])

fn_1_args = [converter.s_exp_to_python_ast(x) for x in ["&a:0", "&z:0"]]

fn_2_body = """
(If
    (Compare (Name %2 Load) (list Eq) (list (Constant i0 None)))
    (/seq
        (If
        (Compare (Name %3 Load) (list Eq) (list (Constant i0 None)))
        (/seq
            (If
            (Compare (Name %1 Load) (list Eq) (list (Constant i0 None)))
            (/seq (Expr (Call (Name g_print Load) (list (_starred_content (UnaryOp USub (Constant i1 None)))) nil)))
            (/seq (Expr (Call (Name g_print Load) (list (_starred_content (Constant i0 None))) nil)))
            )
        )
        (/seq
            (Expr (Call (Name g_print Load) (list (_starred_content (Constant i1 None))) nil))
            (Expr (Call (Name g_print Load) (list (_starred_content (BinOp (UnaryOp USub (Name %1 Load)) Div (Name %3 Load)))) nil))
        )
        )
    )
    (/seq
        ?0
        (Assign
        (list (Name %4 Store))
        (BinOp
            (BinOp (Name %3 Load) Pow (Constant i2 None))
            Sub
            (BinOp (BinOp (Constant i4 None) Mult (Name %2 Load)) Mult (Name %1 Load))
        )
        None
        )
        (If
        (Compare (Name %4 Load) (list Gt) (list (Constant i0 None)))
        (/seq (Expr #0))
        (/seq
            (If
            (Compare (Name %4 Load) (list Eq) (list (Constant i0 None)))
            (/seq
                (Expr (Call (Name g_print Load) (list (_starred_content (Constant i1 None))) nil))
                (Expr
                (Call
                    (Name g_print Load)
                    (list (_starred_content (BinOp (UnaryOp USub (Name %3 Load)) Div (BinOp (Constant i2 None) Mult (Name %2 Load)))))
                    nil
                )
                )
            )
            (/seq (Expr (Call (Name g_print Load) (list (_starred_content (Constant i0 None))) nil)))
            )
        )
        )
    )
)
"""

fn_2 = Abstraction.of(
    "fn_2",
    fn_2_body,
    "S",
    dfa_symvars=["X", "X", "X", "X"],
    dfa_metavars=["E"],
    dfa_choicevars=["seqS"],
)

fn_2_args_w_nothing = [
    converter.s_exp_to_python_ast(x)
    for x in [
        "(Call (Name g_print Load) (list (Constant i2 None)) nil)",
        "&c:0",
        "&a:0",
        "&b:0",
        "&d:0",
        "(/choiceseq)",
    ]
]
fn_2_args = fn_2_args_w_nothing[:-1] + [
    ns.python_statements_to_python_ast("if x == 3: pass")
]
fn_2_args_with_stub = fn_2_args_w_nothing[:-1] + [
    converter.s_exp_to_python_ast("(fn_3)")
]

fn_3 = Abstraction.of(
    "fn_3",
    """
        (/seq
            (Assign (list (Name &x:0 Store)) (Constant i30 None) None)
            (Assign (list (Name &x:0 Store)) (Constant i10 None) None))
        """,
    dfa_root="seqS",
)


def assertSameCode(test, actual, expected):
    print(actual)
    test.assertEqual(
        dedent(actual).strip(),
        dedent(expected).strip(),
    )


class AbstractionRenderingTest(unittest.TestCase):

    def test_stub_rendering_simple(self):
        stub = fn_1.create_stub(fn_1_args, is_pythonm=False)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_1(__ref__(a), __ref__(z))
            """,
        )

        stub = fn_1.create_stub(fn_1_args, is_pythonm=True)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_1(&a, &z)
            """,
        )

    def test_stub_rendering_multi(self):
        stub = fn_2.create_stub(fn_2_args, is_pythonm=False)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(__code__('print(2)'), __ref__(c), __ref__(a), __ref__(b), __ref__(d), __code__('if x == 3:\\n    pass'))
            """,
        )

        stub = fn_2.create_stub(fn_2_args, is_pythonm=True)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(`print(2)`, &c, &a, &b, &d, `if x == 3:\\n    pass`)
            """,
        )

    def test_stub_rendering_multi_w_nothing(self):
        stub = fn_2.create_stub(fn_2_args_w_nothing, is_pythonm=False)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(__code__('print(2)'), __ref__(c), __ref__(a), __ref__(b), __ref__(d), __code__(''))
            """,
        )

        stub = fn_2.create_stub(fn_2_args_w_nothing, is_pythonm=True)

        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(`print(2)`, &c, &a, &b, &d, ``)
            """,
        )

    def test_body_rendering_simple(self):
        stub = fn_1.substitute_body(fn_1_args)
        assertSameCode(
            self,
            stub.to_python(),
            """
            a = int(input())
            z = input()
            """,
        )

    def test_body_rendering_simple_with_pragmas(self):
        stub = fn_1.substitute_body(fn_1_args, pragmas=True)
        assertSameCode(
            self,
            stub.to_python(),
            """
            __start_extract__
            a = int(input())
            z = input()
            __end_extract__
            """,
        )

    def test_body_rendering_multi(self):
        stub = fn_2.substitute_body(fn_2_args)
        print(stub.to_python())
        assertSameCode(
            self,
            stub.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                if x == 3:
                    pass
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )

    def test_body_rendering_with_choiceseq_abstraction(self):
        element = fn_2.substitute_body(fn_2_args_with_stub)
        stub = replace_abstraction_calls(
            element,
            {
                k: fn_3.create_stub([], is_pythonm=False)
                for k in collect_abstraction_calls(element)
            },
        )
        assertSameCode(
            self,
            stub.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                fn_3()
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )
        body = abstraction_calls_to_bodies(element, {"fn_3": fn_3})
        assertSameCode(
            self,
            body.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                x = 30
                x = 10
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )

    def test_body_expand_recursive(self):
        context = converter.s_exp_to_python_ast(
            "(fn_2 (Call (Name g_print Load) (list (Constant i2 None)) nil) &c:0 &a:0 &b:0 &d:0 (/choiceseq (fn_3)))"
        )
        context = abstraction_calls_to_bodies_recursively(
            context, {"fn_2": fn_2, "fn_3": fn_3}
        )
        assertSameCode(
            self,
            context.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                x = 30
                x = 10
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )

    def test_body_expand_recursive_two(self):
        context = converter.s_exp_to_python_ast(
            "(/seq (fn_2 (Call (Name g_print Load) (list (Constant i2 None)) nil) &c:0 &a:0 &b:0 &d:0 (/choiceseq (fn_3))) (fn_2 (Constant i10 None) &c:0 &a:0 &b:0 &d:0 (/choiceseq (fn_3))))"
        )
        context = abstraction_calls_to_bodies_recursively(
            context, {"fn_2": fn_2, "fn_3": fn_3}
        )
        assertSameCode(
            self,
            context.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                x = 30
                x = 10
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                x = 30
                x = 10
                d = b ** 2 - 4 * a * c
                if d > 0:
                    10
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )

    def test_body_expanded_twice(self):
        abstractions = {
            "fn_0": Abstraction.of(
                **{
                    "name": "fn_0",
                    "body": "(Expr (Call (Name g_func Load) (list (_starred_content (BinOp (BinOp #0 Add #0) Add (Constant i0 None)))) nil))",
                    "dfa_root": "S",
                    "dfa_symvars": [],
                    "dfa_metavars": ["E"],
                    "dfa_choicevars": [],
                }
            ),
            "fn_1": Abstraction.of(
                **{
                    "name": "fn_1",
                    "body": "(fn_0 #0)",
                    "arity": 1,
                    "sym_arity": 0,
                    "choice_arity": 0,
                    "dfa_root": "S",
                    "dfa_symvars": [],
                    "dfa_metavars": ["E"],
                    "dfa_choicevars": [],
                }
            ),
        }
        program = converter.s_exp_to_python_ast(
            "(Module (/seq (fn_1 (Name g_e Load)) (fn_1 (Name g_g Load))) nil)"
        )
        code = abstraction_calls_to_bodies_recursively(program, abstractions)
        self.assertEqual(code.to_python(), "func(e + e + 0)\nfunc(g + g + 0)")

    def test_expand_unique_ids(self):
        # This test checks that ids are reset to be unique when a body containing a stub is expanded.
        # Related to test_body_expanded_twice.
        abstractions = {
            "fn_1": Abstraction.of(
                **{
                    "name": "fn_1",
                    "body": "(fn_0 #0)",
                    "arity": 1,
                    "sym_arity": 0,
                    "choice_arity": 0,
                    "dfa_root": "S",
                    "dfa_symvars": [],
                    "dfa_metavars": ["E"],
                    "dfa_choicevars": [],
                }
            ),
        }
        program = converter.s_exp_to_python_ast(
            "(Module (/seq (fn_1 (Name g_e Load)) (fn_1 (Name g_g Load))) nil)"
        )
        code = abstraction_calls_to_bodies(program, abstractions)
        assert code.typ == ast.Module
        code = code.children[0]
        assert isinstance(code, ns.SequenceAST)
        [e_call, g_call] = code.elements
        [e] = e_call.args
        [g] = g_call.args
        self.assertEqual(e.to_python(), "e")
        self.assertEqual(g.to_python(), "g")
        self.assertNotEqual(e_call.handle, g_call.handle)

    def test_body_rendering_multi_with_pragmas(self):
        stub = fn_2.substitute_body(fn_2_args, pragmas=True)
        print(stub.to_python())
        assertSameCode(
            self,
            stub.to_python(),
            """
            __start_extract__
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                __start_choice__
                if x == 3:
                    pass
                __end_choice__
                d = b ** 2 - 4 * a * c
                if d > 0:
                    {__metavariable__, __m0, print(2)}
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            __end_extract__
            """,
        )

    def test_body_rendering_multi_w_nothing(self):
        stub = fn_2.substitute_body(fn_2_args_w_nothing)
        assertSameCode(
            self,
            stub.to_python(),
            """
            if a == 0:
                if b == 0:
                    if c == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-c / b)
            else:
                d = b ** 2 - 4 * a * c
                if d > 0:
                    print(2)
                elif d == 0:
                    print(1)
                    print(-b / (2 * a))
                else:
                    print(0)
            """,
        )

    def test_body_variable_rendering_simple(self):
        body = fn_1.body_with_variable_names()
        assertSameCode(
            self,
            body.to_python(),
            """
            %1 = int(input())
            %2 = input()
            """,
        )

    def test_body_variable_rendering_multi(self):
        body = fn_2.body_with_variable_names()
        self.maxDiff = None
        assertSameCode(
            self,
            body.to_python(),
            """
            if %2 == 0:
                if %3 == 0:
                    if %1 == 0:
                        print(-1)
                    else:
                        print(0)
                else:
                    print(1)
                    print(-%1 / %3)
            else:
                ?0
                %4 = %3 ** 2 - 4 * %2 * %1
                if %4 > 0:
                    #0
                elif %4 == 0:
                    print(1)
                    print(-%3 / (2 * %2))
                else:
                    print(0)
            """,
        )

    def test_nested_abstraction_render(self):
        body = """
        (/subseq
            (While
                (Name %2 Load)
                (/seq
                    (AugAssign
                        (Name %3 Store)
                        Add
                        (BinOp (Name %2 Load) Mod (Name %1 Load)))
                    (AugAssign (Name %2 Store) FloorDiv (Name %1 Load))
                    ?0)
                (/seq))
            (If
                (Compare
                    (Name %3 Load) (list Eq) (list (Name %4 Load)))
                (/seq (Return (Name %1 Load)))
                (/seq)))
        """

        fn_5 = Abstraction.of(
            "fn_5",
            body,
            "seqS",
            dfa_symvars=["Name", "Name", "Name", "Name"],
            dfa_choicevars=["seqS"],
        )
        tmp_abstraction_calls = {"fn_5": fn_5}
        result = ns.render_s_expression(
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast("(/splice (fn_5 %1 %4 %5 %2 #0))"),
                tmp_abstraction_calls,
            ).to_ns_s_exp()
        )
        expected = """
        (/splice
            (/subseq
                (While
                    (Name %4 Load)
                    (/seq
                        (AugAssign
                            (Name %5 Store)
                            Add
                            (BinOp (Name %4 Load) Mod (Name %1 Load)))
                        (AugAssign (Name %4 Store) FloorDiv (Name %1 Load))
                        (/splice #0))
                        (/seq))
                    (If
                        (Compare (Name %5 Load) (list Eq) (list (Name %2 Load)))
                        (/seq (Return (Name %1 Load))) (/seq))))
        """
        expected = ns.render_s_expression(ns.parse_s_expression(expected))
        self.assertEqual(result, expected)

    def test_dfa_with_abstractions_works(self):
        export_dfa(abstrs={"fn_1": fn_1, "fn_2": fn_2})

    def test_dsl_with_abstractions_works(self):
        dfa = export_dfa(abstrs={"fn_1": fn_1, "fn_2": fn_2})
        subset = ns.PythonDSLSubset.from_programs(
            dfa,
            ns.python_to_python_ast("x = x + 2; y = y + x + 2"),
            root="M",
        )
        ns.create_python_dsl(dfa, subset, "M")

    def test_in_order_simple(self):
        self.assertEqual(
            fn_1.variables_in_order(
                ns.python_def_use_mask.python_ordering_dictionary()
            ),
            ["%1", "%2"],
        )
        self.assertEqual(
            fn_1.arguments_traversal_order(
                ns.python_def_use_mask.python_ordering_dictionary()
            ),
            [0, 1],
        )

    def test_in_order_multi(self):
        self.assertEqual(
            fn_2.variables_in_order(
                ns.python_def_use_mask.python_ordering_dictionary()
            ),
            ["%2", "%3", "%1", "?0", "%4", "#0"],
        )
        # order is #0 %1 %2 %3 %4 ?0
        self.assertEqual(
            fn_2.arguments_traversal_order(
                ns.python_def_use_mask.python_ordering_dictionary()
            ),
            [2, 3, 1, 5, 4, 0],
        )

    def test_in_order_comprehension(self):
        fn = Abstraction.of(
            "fn_3",
            """
            (Expr
                (ListComp
                    #0
                    (list
                        (comprehension
                            (Name %1 Store)
                            (Call 
                                #1
                                (list (_starred_content (Constant i10 None))) 
                                nil)
                            nil
                            i0))))
            """,
            "S",
            dfa_symvars=["Name"],
            dfa_metavars=["E", "E"],
        )
        self.assertEqual(
            fn.variables_in_order(ns.python_def_use_mask.python_ordering_dictionary()),
            ["%1", "#1", "#0"],
        )

    def check_abstraction_bodies_in(self, x):
        x = copy.deepcopy(x)
        abstractions = [
            Abstraction.of(**abstraction, name=f"fn_{idx}")
            for idx, abstraction in enumerate(x["abstractions"], 1)
        ]
        python_node_ordering_with_abstractions(abstractions)

    @parameterized.expand(range(len(load_stitch_output_set())))
    def test_abstraction_bodies_in_order_no_crash(self, i):
        self.check_abstraction_bodies_in(load_stitch_output_set()[i])

    @parameterized.expand(range(len(load_stitch_output_set_no_dfa())))
    def test_abstraction_bodies_in_order_no_crash_no_dfa(self, i):
        self.check_abstraction_bodies_in(load_stitch_output_set_no_dfa()[i])


class AbstractionRenderingAnnieSetTest(unittest.TestCase):
    def check_renders(self, s_exp):
        print(s_exp)
        parsed = converter.s_exp_to_python_ast(s_exp)
        print(parsed)
        self.assertEqual(ns.render_s_expression(parsed.to_ns_s_exp()), s_exp)

    def check_renders_with_bodies_expanded(self, s_exp, abstrs):
        abstrs_dict = {x.name: x for x in abstrs}
        parsed = converter.s_exp_to_python_ast(s_exp)
        parsed = abstraction_calls_to_bodies_recursively(parsed, abstrs_dict)
        parsed.to_ns_s_exp()
        parsed.to_python()

    @expand_with_slow_tests(len(load_annies_compressed_individual_programs()), 10)
    def test_renders_realistic(self, i):
        _, rewritten = load_annies_compressed_individual_programs()[i]
        self.check_renders(rewritten)

    @expand_with_slow_tests(len(load_annies_compressed_individual_programs()), 10)
    def test_renders_realistic_with_bodies_expanded(self, i):
        abstractions, rewritten = load_annies_compressed_individual_programs()[i]
        self.check_renders_with_bodies_expanded(rewritten, abstractions)
