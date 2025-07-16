import copy
import unittest
from textwrap import dedent

from permacache import permacache, stable_hash

from imperative_stitch.analyze_program.extract.errors import NotApplicable
from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies,
    abstraction_calls_to_bodies_recursively,
    abstraction_calls_to_stubs,
    collect_abstraction_calls,
    replace_abstraction_calls,
)
from imperative_stitch.compress.run_extraction import convert_output
from imperative_stitch.data.stitch_output_set import (
    load_stitch_output_set,
    load_stitch_output_set_no_dfa,
)
from imperative_stitch.parser import converter
from imperative_stitch.parser.python_ast import AbstractionCallAST
from imperative_stitch.utils.run_code import run_python_with_timeout
from tests.utils import expand_with_slow_tests


def assertSameCode(test, actual, expected):
    actual = dedent(actual).strip()
    print(actual)
    test.assertEqual(
        actual,
        dedent(expected).strip(),
    )


class SequenceTest(unittest.TestCase):
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

    fn_1_body = """
    (/subseq
        (Assign (list (Name %1 Store)) (Call (Name g_int Load) (list (Call (Name g_input Load) nil nil)) nil) None)
        (Assign (list (Name %2 Store)) (Call (Name g_input Load) nil nil) None))
    """

    fn_1 = Abstraction.of("fn_1", fn_1_body, "seqS", dfa_symvars=["X", "X"])

    abtractions = {"fn_1": fn_1}

    def test_stub_insertion_subseq(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_in_seq), self.abtractions
            ).to_python(),
            """
            fn_1(__ref__(n), __ref__(s))
            k = s.count('8')
            """,
        )

    def test_stub_insertion_rooted(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_rooted), self.abtractions
            ).to_python(),
            """
            if x:
                fn_1(__ref__(a), __ref__(z))
            """,
        )

    def test_stub_insertion_rooted_substitute_variables(self):
        parsed = converter.s_exp_to_python_ast(self.ctx_rooted)
        abstracts = collect_abstraction_calls(parsed)
        # pylint: disable=unbalanced-dict-unpacking
        [handle] = abstracts.keys()
        ac = abstracts[handle]
        new_abstraction_call = AbstractionCallAST(
            tag=ac.tag,
            handle=ac.handle,
            args=[
                converter.s_exp_to_python_ast(x)
                for x in [
                    "&u:0",
                    "&v:0",
                ]
            ],
        )
        parsed = replace_abstraction_calls(parsed, {handle: new_abstraction_call})
        assertSameCode(
            self,
            abstraction_calls_to_stubs(parsed, self.abtractions).to_python(),
            """
            if x:
                fn_1(__ref__(u), __ref__(v))
            """,
        )

    def test_injection_subseq(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_in_seq), self.abtractions
            ).to_python(),
            """
            n = int(input())
            s = input()
            k = s.count('8')
            """,
        )

    def test_injection_subseq_pragmas(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_in_seq),
                self.abtractions,
                pragmas=True,
            ).to_python(),
            """
            __start_extract__
            n = int(input())
            s = input()
            __end_extract__
            k = s.count('8')
            """,
        )

    def test_injection_rooted(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_rooted), self.abtractions
            ).to_python(),
            """
            if x:
                a = int(input())
                z = input()
            """,
        )

    def test_injection_rooted_pragmas(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_rooted),
                self.abtractions,
                pragmas=True,
            ).to_python(),
            """
            if x:
                __start_extract__
                a = int(input())
                z = input()
                __end_extract__
            """,
        )


class MultiKindTest(unittest.TestCase):
    fn_1 = Abstraction.of(
        "fn_1",
        "(/seq (If (BinOp (BinOp (BinOp (BinOp (Constant i1 None) Mult (Constant i2 None)) Mult (Constant i3 None)) Mult (Constant i4 None)) Mult (Constant i5 None)) (/seq (Assign (list (Name %1 Store)) (BinOp (Name %1 Load) Add #0) None) (Assign (list (Name %2 Store)) (List (list (Name g_print Load) (Name g_sum Load) (Name g_u Load)) Load) None) ?0) nil))",
        "seqS",
        dfa_symvars=["X", "X"],
        dfa_metavars=["E"],
        dfa_choicevars=["seqS"],
    )

    ctx_includes_choicevar = """
    (Module
        (fn_1
            (BinOp (Constant i1 None) Mult (Constant i2 None))
            &x:0
            &y:0
            (/choiceseq (Assign (list (Name &z:0 Store)) (Name &x:0 Load) None)))
        nil)
    """

    ctx_includes_metavariable_stub = """
    (Module
        (fn_1
            (fn_5)
            &x:0
            &y:0
            (/choiceseq (Assign (list (Name &z:0 Store)) (Name &x:0 Load) None)))
        nil)
    """

    ctx_includes_choicevar_stub = """
    (Module
        (fn_1
            (BinOp (Constant i1 None) Mult (Constant i2 None))
            &x:0
            &y:0
            (/choiceseq (fn_6)))
        nil)
    """

    ctx_includes_choicevar_seq_stub = """
    (Module
        (fn_1
            (BinOp (Constant i1 None) Mult (Constant i2 None))
            &x:0
            &y:0
            (fn_6))
        nil)
    """

    ctx_includes_multi_choicevar = """
    (Module
        (fn_1
            (BinOp (Constant i1 None) Mult (Constant i2 None))
            &x:0
            &y:0
            (/choiceseq
                (Assign (list (Name &z:0 Store)) (Name &x:0 Load) None)
                (Assign (list (Name &u:0 Store)) (Name &x:0 Load) None)))
        nil)
    """

    ctx_no_choicevar = """
    (Module
        (fn_1
            (BinOp (Constant i4 None) Mult (Constant i3 None))
            &x:0
            &y:0 
            (/choiceseq))
        nil)
    """

    fn_2 = Abstraction.of(
        "fn_2",
        """
        (If
            (Compare #1 (list Eq) (list (Constant i0 None)))
            (/seq (Expr (Call (Name g_print Load) (list #2) nil)))
            (/seq
                (If
                    (Compare #1 (list Eq) (list (Constant i1 None)))
                    (/seq (Expr (Call (Name g_print Load) (list #0) nil)))
                    (/seq (Expr (Call (Name g_print Load) (list (Constant i0 None)) nil)))
                )
            )
        )
        """,
        "S",
        dfa_metavars=["E", "E", "E"],
    )

    ctx_for_fn2_1 = """
    (Module
        (/seq
            (Assign
                (list (Name &n:0 Store))
                (Call (Name g_int Load) (list (Call (Name g_input Load) nil nil)) nil)
                None
            )
            (Assign (list (Name &c:0 Store)) (Call (Name g_input Load) nil nil) None)
            (fn_2
                (Constant i1 None)
                (Call (Attribute (Name &c:0 Load) s_count Load) (list (Constant s_I None)) nil)
                (Call (Attribute (Name &c:0 Load) s_count Load) (list (Constant s_A None)) nil)
            )
        )
        nil
    )
    """

    fn_3 = Abstraction.of(
        "fn_3",
        "(BinOp (BinOp #0 Mult (BinOp #0 Add (Constant i1 None))) FloorDiv (Constant i2 None))",
        "E",
        dfa_metavars=["E"],
    )

    ctx_for_fn3_1 = """
    (Module
        (/seq
            (Assign
                (list (Name &a:0 Store))
                (Call (Name g_int Load) (list (Call (Name g_input Load) nil nil)) nil)
                None
            )
            (Assign
                (list (Name &b:0 Store))
                (Call (Name g_int Load) (list (Call (Name g_input Load) nil nil)) nil)
                None
            )
            (Assign
                (list (Name &mid:0 Store))
                (BinOp (BinOp (Name &a:0 Load) Add (Name &b:0 Load)) FloorDiv (Constant i2 None))
                None
            )
            (Assign
                (list (Name &a:0 Store))
                (Call (Name g_abs Load) (list (BinOp (Name &mid:0 Load) Sub (Name &a:0 Load))) nil)
                None
            )
            (Assign
                (list (Name &b:0 Store))
                (Call (Name g_abs Load) (list (BinOp (Name &mid:0 Load) Sub (Name &b:0 Load))) nil)
                None
            )
            (Expr
                (Call
                    (Name g_print Load)
                    (list (BinOp (fn_3 (Name &a:0 Load)) Add (fn_3 (Name &b:0 Load))))
                    nil
                )
            )
        )
        nil
    )
    """

    fn_4 = Abstraction.of(
        "fn_4",
        "(/seq (If #0 (/seq (Expr (Constant True None))) (/seq ?0)))",
        "seqS",
        dfa_symvars=["X", "X"],
        dfa_metavars=["E"],
        dfa_choicevars=["seqS"],
    )

    fn_5 = Abstraction.of(
        "fn_5",
        "(Assign (list (Name &x:0 Store)) (Call (Name g_input Load) nil nil) None)",
        "S",
    )

    fn_6 = Abstraction.of(
        "fn_6",
        "(/seq (Assign (list (Name &z:0 Store)) (Name &x:0 Load) None))",
        "seqS",
    )

    abstractions = {
        "fn_1": fn_1,
        "fn_2": fn_2,
        "fn_3": fn_3,
        "fn_4": fn_4,
        "fn_5": fn_5,
        "fn_6": fn_6,
    }

    def test_stub_includes_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_includes_choicevar),
                self.abstractions,
            ).to_python(),
            """
            fn_1(__code__('1 * 2'), __ref__(x), __ref__(y), __code__('z = x'))
            """,
        )

    def test_stub_includes_multi_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_includes_multi_choicevar),
                self.abstractions,
            ).to_python(),
            r"""
            fn_1(__code__('1 * 2'), __ref__(x), __ref__(y), __code__('z = x\nu = x'))
            """,
        )

    def test_stub_no_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_no_choicevar), self.abstractions
            ).to_python(),
            """
            fn_1(__code__('4 * 3'), __ref__(x), __ref__(y), __code__(''))
            """,
        )

    def test_stub_metavariable_stub(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_includes_metavariable_stub),
                self.abstractions,
            ).to_python(),
            """
            fn_1(__code__('fn_5()'), __ref__(x), __ref__(y), __code__('z = x'))
            """,
        )

    def test_stub_choicevar_stub(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_includes_choicevar_stub),
                self.abstractions,
            ).to_python(),
            """
            fn_1(__code__('1 * 2'), __ref__(x), __ref__(y), __code__('fn_6()'))
            """,
        )

    def test_stub_choicevar_seq_stub(self):
        assertSameCode(
            self,
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(self.ctx_includes_choicevar_seq_stub),
                self.abstractions,
            ).to_python(),
            """
            fn_1(__code__('1 * 2'), __ref__(x), __ref__(y), __code__('fn_6()'))
            """,
        )

    def test_injection_includes_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_includes_choicevar),
                self.abstractions,
            ).to_python(),
            """
            if 1 * 2 * 3 * 4 * 5:
                x = x + 1 * 2
                y = [print, sum, u]
                z = x
            """,
        )

    def test_rooted_choicevar_body(self):
        res = abstraction_calls_to_bodies(
            converter.s_exp_to_python_ast(
                self.ctx_includes_choicevar.replace("fn_1", "fn_4")
            ),
            self.abstractions,
        )
        print(res)
        assertSameCode(
            self,
            res.to_python(),
            """
            if 1 * 2:
                True
            else:
                z = x
            """,
        )

    def test_rooted_choicevar_body_missing(self):
        res = abstraction_calls_to_bodies(
            converter.s_exp_to_python_ast(
                self.ctx_no_choicevar.replace("fn_1", "fn_4")
            ),
            self.abstractions,
        )
        print(res)
        assertSameCode(
            self,
            res.to_python(),
            """
            if 4 * 3:
                True
            """,
        )

    def test_injection_includes_choicevar_pragmas(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_includes_choicevar),
                self.abstractions,
                pragmas=True,
            ).to_python(),
            """
            __start_extract__
            if 1 * 2 * 3 * 4 * 5:
                x = x + {__metavariable__, __m0, 1 * 2}
                y = [print, sum, u]
                __start_choice__
                z = x
                __end_choice__
            __end_extract__
            """,
        )

    def test_injection_includes_multi_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_includes_multi_choicevar),
                self.abstractions,
            ).to_python(),
            """
            if 1 * 2 * 3 * 4 * 5:
                x = x + 1 * 2
                y = [print, sum, u]
                z = x
                u = x
            """,
        )

    def test_injection_doesnt_include_choicevar(self):
        assertSameCode(
            self,
            abstraction_calls_to_bodies(
                converter.s_exp_to_python_ast(self.ctx_no_choicevar), self.abstractions
            ).to_python(),
            """
            if 1 * 2 * 3 * 4 * 5:
                x = x + 4 * 3
                y = [print, sum, u]
            """,
        )

    def test_stub_fn2_1(self):
        out = abstraction_calls_to_stubs(
            converter.s_exp_to_python_ast(self.ctx_for_fn2_1), self.abstractions
        ).to_python()
        self.maxDiff = None
        assertSameCode(
            self,
            out,
            r"""
            n = int(input())
            c = input()
            fn_2(__code__('1'), __code__("c.count('I')"), __code__("c.count('A')"))
            """,
        )

    def test_injection_fn2_1(self):
        out = abstraction_calls_to_bodies(
            converter.s_exp_to_python_ast(self.ctx_for_fn2_1), self.abstractions
        ).to_python()
        self.maxDiff = None
        assertSameCode(
            self,
            out,
            r"""
            n = int(input())
            c = input()
            if c.count('I') == 0:
                print(c.count('A'))
            elif c.count('I') == 1:
                print(1)
            else:
                print(0)
            """,
        )

    def test_injection_fn2_1_pragmas(self):
        out = abstraction_calls_to_bodies(
            converter.s_exp_to_python_ast(self.ctx_for_fn2_1),
            self.abstractions,
            pragmas=True,
        ).to_python()
        self.maxDiff = None
        assertSameCode(
            self,
            out,
            r"""
            n = int(input())
            c = input()
            __start_extract__
            if {__metavariable__, __m1, c.count('I')} == 0:
                print({__metavariable__, __m2, c.count('A')})
            elif {__metavariable__, __m1, c.count('I')} == 1:
                print({__metavariable__, __m0, 1})
            else:
                print(0)
            __end_extract__
            """,
        )

    def test_stub_fn3_1(self):
        out = abstraction_calls_to_stubs(
            converter.s_exp_to_python_ast(self.ctx_for_fn3_1), self.abstractions
        ).to_python()
        self.maxDiff = None
        assertSameCode(
            self,
            out,
            r"""
            a = int(input())
            b = int(input())
            mid = (a + b) // 2
            a = abs(mid - a)
            b = abs(mid - b)
            print(fn_3(__code__('a')) + fn_3(__code__('b')))
            """,
        )

    def test_injection_fn3_1(self):
        out = abstraction_calls_to_bodies(
            converter.s_exp_to_python_ast(self.ctx_for_fn3_1), self.abstractions
        ).to_python()
        self.maxDiff = None
        assertSameCode(
            self,
            out,
            r"""
            a = int(input())
            b = int(input())
            mid = (a + b) // 2
            a = abs(mid - a)
            b = abs(mid - b)
            print(a * (a + 1) // 2 + b * (b + 1) // 2)
            """,
        )


class RealDataTest(unittest.TestCase):
    def check_is_parseable(self, eg, check_stubs_pragmas=True):
        eg = copy.deepcopy(eg)
        abstr = {
            f"fn_{i}": Abstraction.of(name=f"fn_{i}", **abstr_dict)
            for i, abstr_dict in enumerate(eg["abstractions"], 1)
        }
        print(abstr)
        for code, rewritten in zip(eg["code"], eg["rewritten"]):
            code = converter.s_exp_to_python(code)
            if check_stubs_pragmas:
                abstraction_calls_to_stubs(
                    converter.s_exp_to_python_ast(rewritten), abstr
                )
            out = abstraction_calls_to_bodies_recursively(
                converter.s_exp_to_python_ast(rewritten), abstr
            ).to_python()
            print("#" * 80)
            print(code)
            print("*" * 80)
            print(out)
            print("#" * 80)
            assertSameCode(
                self,
                out,
                code,
            )
            if check_stubs_pragmas:
                check_no_crash = abstraction_calls_to_bodies(
                    converter.s_exp_to_python_ast(rewritten), abstr, pragmas=True
                ).to_python()
                self.assertIsNotNone(check_no_crash)

    @expand_with_slow_tests(len(load_stitch_output_set()))
    def test_realistic_parseable(self, i):
        self.check_is_parseable(load_stitch_output_set()[i])

    @expand_with_slow_tests(len(load_stitch_output_set_no_dfa()))
    def test_realistic_parseable_no_dfa(self, i):
        self.check_is_parseable(
            load_stitch_output_set_no_dfa()[i], check_stubs_pragmas=False
        )

    def currently_invalid(self, abstrs):
        return any(abstr["dfa_choicevars"] for abstr in abstrs)

    def check_same_behavior(self, eg):
        eg = copy.deepcopy(eg)
        if self.currently_invalid(eg["abstractions"]):
            return
        try:
            abstraction, rewritten = convert_output(eg["abstractions"], eg["rewritten"])
        except NotApplicable:
            # This is fine, we can't rewrite this example
            return
        from ..extract.rewrite_semantic_test import RewriteSemanticsTest

        assert len(rewritten) == len(eg["code"])

        for rewr, code_original in zip(rewritten, eg["code"]):
            code_original = converter.s_exp_to_python(code_original)
            print(code_original)
            out = outputs(code_original, eg["inputs"][:10])
            if out is None:
                continue
            RewriteSemanticsTest().assert_code_same(
                dict(
                    inputs=eg["inputs"][:10],
                    outputs=out,
                ),
                code_original,
                rewr,
                extracted=abstraction,
            )

    @expand_with_slow_tests(len(load_stitch_output_set()))
    def test_realistic_same_behavior(self, i):
        self.check_same_behavior(load_stitch_output_set()[i])


@permacache(
    "imperative_stitch/tests/from_stitch_test/outputs",
    key_function=dict(code=stable_hash, inputs=stable_hash),
    multiprocess_safe=True,
)
def outputs(code, inputs):
    result = []
    for inp in inputs:
        out = run_python_with_timeout(code, inp, timeout=1)
        if out is None:
            return None
        result.append(out)
    return result
