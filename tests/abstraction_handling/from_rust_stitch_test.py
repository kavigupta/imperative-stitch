import ast
from textwrap import dedent
import unittest
import stitch_core
import neurosym as ns

from imperative_stitch.compress.manipulate_abstraction import abstraction_calls_to_stubs
from imperative_stitch.compress.rust_stitch import process_rust_stitch
from imperative_stitch.parser import converter
from tests.utils import canonicalize


def compress_stitch(pythons, **kwargs) -> stitch_core.CompressionResult:
    s_exps = [ns.python_to_s_exp(code_snippet) for code_snippet in pythons]
    print("s_exps", s_exps)
    compressed = stitch_core.compress(
        s_exps,
        cost_prim='{"Module":0,"Name":0,"Load":0,"Store":0,"None":0,"list":0,"nil":0,"semi":0,"Constant":0,"Attribute":0,"_slice_content":0,"_slice_slice":0,"_slice_tuple":0,"_starred_content":0,"_starred_starred":0,"/choiceseq":0,"Subscript":0,"Expr":0,"Call":0,"Assign":0,"BinOp":0}',
        tdfa_json_path="../Stitch.jl/data_for_testing/dfa_imp.json",
        tdfa_root="M",
        valid_metavars='["S","E","seqS"]',
        valid_roots='["S","E","seqS"]',
        tdfa_non_eta_long_states='{"seqS":"S"}',
        symvar_prefix="&",
        **kwargs,
    )
    print("compressed", compressed.abstractions)
    return process_rust_stitch(compressed)


def run_compression_for_testing(code, **kwargs):
    result = compress_stitch(code, **kwargs)
    abstr_dict = {x.name: x for x in result.abstractions}
    abstractions = [
        abstraction_calls_to_stubs(x.body_with_variable_names(), abstr_dict)
        for x in result.abstractions
    ]
    rewritten = [converter.s_exp_to_python_ast(x) for x in result.rewritten]
    rewritten = [abstraction_calls_to_stubs(x, abstr_dict) for x in rewritten]
    rewritten = [x.to_python() for x in rewritten]
    return result.abstractions, [x.to_python() for x in abstractions], rewritten


class TestConversion(unittest.TestCase):

    def test_metavar_symvar_single(self):
        code = [
            dedent(
                """
                x = y = z = 0
                x + y + 4 + z + 2 + 3
                """
            ),
            dedent(
                """
                a = 0
                b = 0
                c = 0
                a + b + 5 + c + 2 + 3
                """
            ),
        ]
        _, abstractions, rewritten = run_compression_for_testing(code, iterations=1)
        self.assertEqual(
            abstractions,
            ["%3 + %2 + #0 + %1 + 2 + 3"],
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize(
                    """
                    x = y = z = 0
                    fn_0(__code__('4'), __ref__(z), __ref__(y), __ref__(x))
                    """
                ),
                canonicalize(
                    """
                    a = 0
                    b = 0
                    c = 0
                    fn_0(__code__('5'), __ref__(c), __ref__(b), __ref__(a))
                    """
                ),
            ],
        )

    def test_metavar_symvar_duplicated(self):
        code = [
            dedent(
                """
                x = y = z = 0
                x + x + x + y + y + z
                """
            ),
            dedent(
                """
                x = 0
                y = 0
                z = 0
                y + y + y + x + x + z
                """
            ),
        ]
        _, abstractions, rewritten = run_compression_for_testing(
            code, iterations=1, max_arity=0
        )
        self.assertEqual(
            abstractions,
            ["%3 + %3 + %3 + %2 + %2 + %1"],
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize(
                    """
                    x = y = z = 0
                    fn_0(__ref__(z), __ref__(y), __ref__(x))
                    """
                ),
                canonicalize(
                    """
                    x = 0
                    y = 0
                    z = 0
                    fn_0(__ref__(z), __ref__(x), __ref__(y))
                    """
                ),
            ],
        )

    def test_metavar_symvar_multi(self):
        code = [
            dedent(
                """
                x = y = z = 0
                x + y + 4 + z + 2 + 3 + 83
                """
            ),
            dedent(
                """
                a = 0
                b = 0
                c = 0
                a + b + 5 + c + 2 + 3 + 83
                a + b + 5 + c + 2 + 3
                a + b + 5 + c + 2 + 3
                """
            ),
        ]
        _, abstractions, rewritten = run_compression_for_testing(code, iterations=2)
        self.assertEqual(
            abstractions,
            [
                "%3 + %2 + #0 + %1 + 2 + 3",
                "fn_0(__code__('#0'), __ref__(%3), __ref__(%2), __ref__(%1)) + 83",
            ],
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize(
                    """
                    x = y = z = 0
                    fn_1(__code__('4'), __ref__(x), __ref__(y), __ref__(z))
                    """
                ),
                canonicalize(
                    """
                    a = 0
                    b = 0
                    c = 0
                    fn_1(__code__('5'), __ref__(a), __ref__(b), __ref__(c))
                    fn_0(__code__('5'), __ref__(c), __ref__(b), __ref__(a))
                    fn_0(__code__('5'), __ref__(c), __ref__(b), __ref__(a))
                    """
                ),
            ],
        )

    def test_sequence_basic(self):
        code = [
            dedent(
                """
                function(x, y, z)
                2 + 3 + 4
                """
            ),
            dedent(
                """
                function(x, y, z2)
                2 + 3 + 4
                """
            ),
            dedent(
                """
                function(x, y, z3)
                2 + 3 + 5
                """
            ),
        ]
        [abstr], [abstraction_text], rewritten = run_compression_for_testing(
            code, iterations=1
        )
        self.assertEqual(
            abstraction_text,
            dedent(
                """
                function(x, y, #1)
                2 + 3 + #0
                """
            ).strip(),
        )
        self.assertEqual(
            abstr.dfa_annotation,
            {"root": "seqS", "metavars": ["E", "E"], "symvars": [], "choicevars": []},
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize("fn_0(__code__('4'), __code__('z'))"),
                canonicalize("fn_0(__code__('4'), __code__('z2'))"),
                canonicalize("fn_0(__code__('5'), __code__('z3'))"),
            ],
        )

    def test_choicevar(self):
        code = [
            dedent(
                """
                if x > 0:
                    y = 2
                    y = 2
                else:
                    y = 3
                """
            ),
            dedent(
                """
                if x > 0:
                    a = 7
                    y = 2
                    y = 2
                else:
                    y = 3
                """
            ),
            dedent(
                """
                if x > 0:
                    f(2)
                    del y
                    y = 2
                else:
                    y = 3
                """
            ),
        ]
        [abstr], [abstraction_text], rewritten = run_compression_for_testing(
            code, iterations=1
        )
        self.assertEqual(
            abstraction_text,
            dedent(
                """
                if x > 0:
                    ?0
                    %1 = 2
                else:
                    %1 = 3
                """
            ).strip(),
        )
        self.assertEqual(
            abstr.dfa_annotation,
            {
                "root": "seqS",
                "metavars": [],
                "symvars": ["Name"],
                "choicevars": ["seqS"],
            },
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize("fn_0(__ref__(y), __code__('y = 2'))"),
                canonicalize("fn_0(__ref__(y), __code__('a = 7\\ny = 2'))"),
                canonicalize("fn_0(__ref__(y), __code__('f(2)\\ndel y'))"),
            ],
        )

    def test_choicevar_in_presence_of_many_metavars(self):
        code = [
            dedent(
                """
                if x > 1000:
                    y = 2
                    y = 23
                else:
                    y = 3
                """
            ),
            dedent(
                """
                if x > 1001:
                    a = 7
                    y = 2
                    y = 24
                else:
                    y = 4
                """
            ),
            dedent(
                """
                if x > 1002:
                    f(2)
                    del y
                    y = 25
                else:
                    y = 5
                """
            ),
        ]
        [abstr], [abstraction_text], rewritten = run_compression_for_testing(
            code, iterations=1, max_arity=10
        )
        self.assertEqual(
            abstraction_text,
            dedent(
                """
                if x > #2:
                    ?0
                    %1 = #1
                else:
                    %1 = #0
                """
            ).strip(),
        )
        self.assertEqual(
            abstr.dfa_annotation,
            {
                "root": "seqS",
                "metavars": ["E", "E", "E"],
                "symvars": ["Name"],
                "choicevars": ["seqS"],
            },
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize(
                    "fn_0(__code__('3'), __code__('23'), __code__('1000'), __ref__(y), __code__('y = 2'))"
                ),
                canonicalize(
                    "fn_0(__code__('4'), __code__('24'), __code__('1001'), __ref__(y), __code__('a = 7\\ny = 2'))"
                ),
                canonicalize(
                    "fn_0(__code__('5'), __code__('25'), __code__('1002'), __ref__(y), __code__('f(2)\\ndel y'))"
                ),
            ],
        )

    def test_choicevar_used_as_rooted(self):
        code = [
            dedent(
                """
                if x > 1000:
                    y = 2
                    y = 23
                else:
                    y = 2
                """
            ),
            dedent(
                """
                if x > 1001:
                    a = 7
                    y = 2
                    y = 24
                else:
                    a = 7
                    y = 2
                """
            ),
            dedent(
                """
                if x > 1002:
                    f(2)
                    del y
                    y = 25
                else:
                    f(2)
                    del y
                """
            ),
        ]
        [abstr], [abstraction_text], rewritten = run_compression_for_testing(
            code, iterations=1, max_arity=10
        )
        self.assertEqual(
            abstraction_text,
            dedent(
                """
                if x > #1:
                    ?0
                    %1 = #0
                else:
                    ?0
                """
            ).strip(),
        )
        self.assertEqual(
            abstr.dfa_annotation,
            {
                "root": "seqS",
                "metavars": ["E", "E"],
                "symvars": ["Name"],
                "choicevars": ["seqS"],
            },
        )
        self.assertEqual(
            rewritten,
            [
                canonicalize(
                    "fn_0(__code__('23'), __code__('1000'), __ref__(y), __code__('y = 2'))"
                ),
                canonicalize(
                    "fn_0(__code__('24'), __code__('1001'), __ref__(y), __code__('a = 7\\ny = 2'))"
                ),
                canonicalize(
                    "fn_0(__code__('25'), __code__('1002'), __ref__(y), __code__('f(2)\\ndel y'))"
                ),
            ],
        )

    # TODO add test where all choicevars have the same length

    # def test_sequence_rooted_not_at_top(self):
    #     code = [
    #         dedent(
    #             """
    #             distraction = 2
    #             function(x, y, z)
    #             2 + 3 + 4
    #             """
    #         ),
    #         dedent(
    #             """
    #             distraction2 * 81
    #             function(x, y, z2)
    #             2 + 3 + 4
    #             """
    #         ),
    #         dedent(
    #             """
    #             1 / distraction3
    #             function(x, y, z3)
    #             2 + 3 + 5
    #             """
    #         ),
    #     ]
    #     [abstr], [abstraction_text], rewritten = run_compression_for_testing(
    #         code, iterations=1
    #     )
    #     self.assertEqual(
    #         abstraction_text,
    #         dedent(
    #             """
    #             function(x, y, #1)
    #             2 + 3 + #0
    #             """
    #         ).strip(),
    #     )
    #     self.assertEqual(
    #         abstr.dfa_annotation,
    #         {"root": "seqS", "metavars": ["E", "E"], "symvars": [], "choicevars": []},
    #     )
    #     self.assertEqual(
    #         rewritten,
    #         [
    #             canonicalize("fn_0(__code__('4'), __code__('z'))"),
    #             canonicalize("fn_0(__code__('4'), __code__('z2'))"),
    #             canonicalize("fn_0(__code__('5'), __code__('z3'))"),
    #         ],
    #     )

    # def test_sequence_rooted_not_at_top_multi(self):
    #     code = [
    #         dedent(
    #             """
    #             abc
    #             distraction = 2
    #             function(x, y, z)
    #             2 + 3 + 4
    #             """
    #         ),
    #         dedent(
    #             """
    #             distraction2 * 81
    #             function(x, y, z2)
    #             2 + 3 + 4
    #             """
    #         ),
    #         dedent(
    #             """
    #             1 / distraction3
    #             function(x, y, z3)
    #             2 + 3 + 5
    #             """
    #         ),
    #     ]
    #     [abstr], [abstraction_text], rewritten = run_compression_for_testing(
    #         code, iterations=1
    #     )
    #     self.assertEqual(
    #         abstraction_text,
    #         dedent(
    #             """
    #             function(x, y, #1)
    #             2 + 3 + #0
    #             """
    #         ).strip(),
    #     )
    #     self.assertEqual(
    #         abstr.dfa_annotation,
    #         {"root": "seqS", "metavars": ["E", "E"], "symvars": [], "choicevars": []},
    #     )
    #     self.assertEqual(
    #         rewritten,
    #         [
    #             canonicalize("fn_0(__code__('4'), __code__('z'))"),
    #             canonicalize("fn_0(__code__('4'), __code__('z2'))"),
    #             canonicalize("fn_0(__code__('5'), __code__('z3'))"),
    #         ],
    #     )
