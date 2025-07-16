import unittest
from textwrap import dedent

import neurosym as ns

from imperative_stitch.compress.rust_stitch import compress_stitch, handle_splice_seqs
from tests.utils import canonicalize, expand_with_slow_tests, small_set_examples


def run_compression_for_testing(*args, **kwargs):
    return TestConversion().run_compression_for_testing(*args, **kwargs)


class TestConversion(unittest.TestCase):

    def run_compression_for_testing(self, code, *, is_pythonm=False, **kwargs):
        result = compress_stitch(code, **kwargs)
        maxDiff, self.maxDiff = self.maxDiff, None
        self.assertEqual(
            [canonicalize(x) for x in code],
            [
                canonicalize(x.to_python())
                for x in result.inline_abstractions(
                    abstraction_names=result.abstr_dict.keys()
                ).rewritten
            ],
        )
        self.maxDiff = maxDiff
        return result

    def test_basic_seq_splice_seq(self):
        self.assertEqual(
            ns.render_s_expression(
                handle_splice_seqs(
                    ns.parse_s_expression("(/seq a b (/splice (/seq 1 2 3)) c d)")
                )
            ),
            "(/seq a b 1 2 3 c d)",
        )
        self.assertEqual(
            ns.render_s_expression(
                handle_splice_seqs(
                    ns.parse_s_expression(
                        "(fn_0 (/seq a b (/splice (/seq 1 2 3)) c d))"
                    )
                )
            ),
            "(fn_0 (/seq a b 1 2 3 c d))",
        )
        self.assertEqual(
            ns.render_s_expression(
                handle_splice_seqs(
                    ns.parse_s_expression(
                        "(fn_0 (/seq a b (/splice (/seq 1 2 3)) c d))"
                    )
                )
            ),
            "(fn_0 (/seq a b 1 2 3 c d))",
        )

    def test_if_with_no_else(self):
        code = [
            dedent(
                """
                if x > 0:
                    y = 2
                """
            ),
            dedent(
                """
                if x > 0:
                    y = 2
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        self.assertEqual(
            result.abstractions_python(),
            ["if x > 0:\n    %1 = 2"],
        )
        self.assertEqual(
            result.rewritten_python(),
            [
                canonicalize("fn_0(__ref__(y))"),
                canonicalize("fn_0(__ref__(y))"),
            ],
        )

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
        result = self.run_compression_for_testing(code, iterations=1)
        self.assertEqual(
            result.abstractions_python(),
            ["%3 + %2 + #0 + %1 + 2 + 3"],
        )
        self.assertEqual(
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=1, max_arity=0)
        self.assertEqual(
            result.abstractions_python(),
            ["%3 + %3 + %3 + %2 + %2 + %1"],
        )
        self.assertEqual(
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=2)
        self.assertEqual(
            result.abstractions_python(),
            [
                "%3 + %2 + #0 + %1 + 2 + 3",
                "fn_0(__code__('#0'), __ref__(%3), __ref__(%2), __ref__(%1)) + 83",
            ],
        )
        self.assertEqual(
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=1)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=1)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
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
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
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

    def test_sequence_rooted_not_at_top(self):
        code = [
            dedent(
                """
                distraction = 2
                function(x, y, z)
                2 + 3 + 4
                """
            ),
            dedent(
                """
                distraction2 * 81
                function(x, y, z2)
                2 + 3 + 5
                """
            ),
            dedent(
                """
                1 / distraction3
                function(x, y, z3)
                2 + 3 + 6
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=3)
        self.maxDiff = None
        rewritten_raw = [
            ns.render_s_expression(x.to_ns_s_exp()) for x in result.rewritten
        ]
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
        self.assertEqual(
            rewritten_raw[0],
            "(Module (/seq (Assign (list (Name &distraction:0 Store)) (Constant i2 None) None) (/splice (fn_0 (Constant i4 None) (Name g_z Load)))) nil)",
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
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    distraction = 2
                    fn_0(__code__('4'), __code__('z'))
                    """
                ),
                canonicalize(
                    """
                    distraction2 * 81
                    fn_0(__code__('5'), __code__('z2'))
                    """
                ),
                canonicalize(
                    """
                    1 / distraction3
                    fn_0(__code__('6'), __code__('z3'))
                    """
                ),
            ],
        )

    def test_sequence_rooted_not_at_top_choicevar(self):
        code = [
            dedent(
                """
                distraction = 2
                x[2] = 4
                function(x, y, z)
                2 + 3 + 4
                """
            ),
            dedent(
                """
                distraction2 * 81
                function(x, y, z2)
                2 + 3 + 5
                """
            ),
            dedent(
                """
                u = 2
                function(x, y, z3)
                2 + 3 + 6
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    distraction = 2
                    x[2] = 4
                    fn_0(__code__('4'), __code__('z'))
                    """
                ),
                canonicalize(
                    """
                    distraction2 * 81
                    fn_0(__code__('5'), __code__('z2'))
                    """
                ),
                canonicalize(
                    """
                    u = 2
                    fn_0(__code__('6'), __code__('z3'))
                    """
                ),
            ],
        )

    def test_sequence_rooted_not_at_top_choicevar_used_internally(self):
        code = [
            dedent(
                """
                distraction = 2
                x[2] = 4
                function(x, y, z)
                2 + 3 + 4
                if x:
                    distraction = 2
                    x[2] = 4

                """
            ),
            dedent(
                """
                distraction2 * 81
                function(x, y, z2)
                2 + 3 + 5
                if x:
                    distraction2 * 81
                """
            ),
            dedent(
                """
                u = 2
                function(x, y, z3)
                2 + 3 + 6
                if x:
                    u = 2
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
        self.assertEqual(
            abstraction_text,
            dedent(
                """
                ?0
                function(x, y, #1)
                2 + 3 + #0
                if x:
                    ?0
                """
            ).strip(),
        )
        self.assertEqual(
            abstr.dfa_annotation,
            {
                "root": "seqS",
                "metavars": ["E", "E"],
                "symvars": [],
                "choicevars": ["seqS"],
            },
        )
        self.assertEqual(
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    fn_0(__code__('4'), __code__('z'), __code__('distraction = 2\\nx[2] = 4'))
                    """
                ),
                canonicalize(
                    """
                    fn_0(__code__('5'), __code__('z2'), __code__('distraction2 * 81'))
                    """
                ),
                canonicalize(
                    """
                    fn_0(__code__('6'), __code__('z3'), __code__('u = 2'))
                    """
                ),
            ],
        )

    def test_sequence_with_suffix(self):
        code = [
            dedent(
                """
                function(x, y, z)
                2 + 3 + 4
                distraction = 2
                x[2] = 4
                """
            ),
            dedent(
                """
                function(x, y, z2)
                2 + 3 + 5
                distraction2 * 81
                """
            ),
            dedent(
                """
                function(x, y, z3)
                2 + 3 + 6
                u = 2
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        [abstraction_text] = result.abstractions_python()
        [abstr] = result.abstractions
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
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    fn_0(__code__('4'), __code__('z'))
                    distraction = 2
                    x[2] = 4
                    """
                ),
                canonicalize(
                    """
                    fn_0(__code__('5'), __code__('z2'))
                    distraction2 * 81
                    """
                ),
                canonicalize(
                    """
                    fn_0(__code__('6'), __code__('z3'))
                    u = 2
                    """
                ),
            ],
        )

    def test_no_seq_seq(self):
        code = [
            canonicalize(
                """
                func
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                a
                b
                c
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=1, max_arity=10)
        for x in [ns.render_s_expression(x.to_ns_s_exp()) for x in result.rewritten]:
            self.assertNotIn("(/seq (/seq", x)

    def test_sequence_is_suffix_of_another_metavar(self):
        code = [
            canonicalize(
                """
                func
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                a
                b
                c
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                2 + 10 ** 27
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                u
                d
                e
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                v = 3
                d
                e
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=2, max_arity=10)
        [abstraction_text1, abstraction_text2] = result.abstractions_python()
        [abstr1, abstr2] = result.abstractions
        self.assertEqual(abstraction_text1, "function(1, 3 ** 2)\n%1 = 2 + 3 + 4")
        self.assertEqual(
            abstr1.dfa_annotation,
            {
                "root": "seqS",
                "metavars": [],
                "symvars": ["Name"],
                "choicevars": [],
            },
        )
        self.assertEqual(abstraction_text2, "d\ne\nfn_0(__ref__(%1))")
        self.assertEqual(
            abstr2.dfa_annotation,
            {
                "root": "seqS",
                "metavars": [],
                "symvars": ["Name"],
                "choicevars": [],
            },
        )
        self.assertEqual(
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    func
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    a
                    b
                    c
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    2 + 10 ** 27
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    u
                    fn_1(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    v = 3
                    fn_1(__ref__(x))
                    """
                ),
            ],
        )

    def test_sequence_is_suffix_of_another_choicevar(self):
        code = [
            canonicalize(
                """
                func
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                a
                b
                c
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                2 + 10 ** 27
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                u
                d
                e
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
            canonicalize(
                """
                w = 7
                v = 3
                d
                e
                function(1, 3 ** 2)
                x = 2 + 3 + 4
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=2, max_arity=10)
        [abstraction_text1, abstraction_text2] = result.abstractions_python()
        [abstr1, abstr2] = result.abstractions
        print("ABC", ns.render_s_expression(abstr2.body.to_ns_s_exp()))
        self.assertEqual(abstraction_text1, "function(1, 3 ** 2)\n%1 = 2 + 3 + 4")
        self.assertEqual(
            abstr1.dfa_annotation,
            {
                "root": "seqS",
                "metavars": [],
                "symvars": ["Name"],
                "choicevars": [],
            },
        )
        self.assertEqual(abstraction_text2, "d\ne\nfn_0(__ref__(%1))")
        self.assertEqual(
            abstr2.dfa_annotation,
            {
                "root": "seqS",
                "metavars": [],
                "symvars": ["Name"],
                "choicevars": [],
            },
        )
        self.assertEqual(
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    func
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    a
                    b
                    c
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    2 + 10 ** 27
                    fn_0(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    u
                    fn_1(__ref__(x))
                    """
                ),
                canonicalize(
                    """
                    w = 7
                    v = 3
                    fn_1(__ref__(x))
                    """
                ),
            ],
        )

    def test_no_abstractions(self):
        code = [
            canonicalize(
                """
                if x:
                    y
                """
            ),
            canonicalize(
                """
                1 + 2
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=3, max_arity=10)
        self.assertEqual(result.rewritten_python(), code)

    def test_empty_exception(self):
        code = [
            canonicalize(
                """
                try:
                    pass
                except:
                    pass
                """
            )
        ]
        result = self.run_compression_for_testing(code, iterations=3, max_arity=10)
        self.assertEqual(code, result.rewritten_python())

    def test_nested_abstractions_multiused(self):
        # See tests/abstraction_handling/abstraction_test.py::AbstractionRenderingTest::test_body_expanded_twice
        code = [
            canonicalize(
                """
                func(a + a + 3)
                func(c + c + 0)
                """
            ),
            canonicalize(
                """
                func(e + e + 0)
                func(g + g + 0)
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=2, max_arity=2)
        [a1, a2] = result.abstractions_python()
        self.assertEqual(
            a1,
            dedent(
                """
                func(#1 + #1 + #0)
                """
            ).strip(),
        )
        self.assertEqual(
            a2,
            dedent(
                """
                fn_0(__code__('0'), __code__('#0'))
                """
            ).strip(),
        )
        self.assertEqual(
            result.rewritten_python(),
            [
                canonicalize(
                    """
                    fn_0(__code__('3'), __code__('a'))
                    fn_1(__code__('c'))
                    """
                ),
                canonicalize(
                    """
                    fn_1(__code__('e'))
                    fn_1(__code__('g'))
                    """
                ),
            ],
        )

    def test_nested_abstractions_inline_one(self):
        code = [
            canonicalize(
                """
                func(a + a + 3)
                func(c + c + 0)
                """
            ),
            canonicalize(
                """
                func(e + e + 0)
                func(g + g + 0)
                """
            ),
        ]
        result = self.run_compression_for_testing(code, iterations=2, max_arity=2)
        result_no_fn_0 = result.inline_abstractions(abstraction_names=["fn_0"])
        [a2] = result_no_fn_0.abstractions_python()
        self.assertEqual(
            a2,
            dedent(
                """
                func(#0 + #0 + 0)
                """
            ).strip(),
        )
        self.assertEqual(
            result_no_fn_0.rewritten_python(),
            [
                canonicalize(
                    """
                    func(a + a + 3)
                    fn_1(__code__('c'))
                    """
                ),
                canonicalize(
                    """
                    fn_1(__code__('e'))
                    fn_1(__code__('g'))
                    """
                ),
            ],
        )

    @expand_with_slow_tests(200, first_fast=3)
    def test_smoke(self, seed):
        if seed == 18 or seed == 102:
            # these take forever. we should look into this.
            return
        programs = small_set_examples()[seed::200]
        # not ascii
        if any(not x.isascii() for x in programs):
            return
        self.run_compression_for_testing(programs, iterations=2, max_arity=1)
