import unittest
from fractions import Fraction

import neurosym as ns
import numpy as np

from imperative_stitch.parser import converter
from imperative_stitch.utils.export_as_dsl import DSLSubset, create_dsl
from tests.dsl_tests.utils import fit_to


class TestLikelihoodFittedDSL(unittest.TestCase):
    def compute_likelihood(self, corpus, program):
        dfa, _, fam, dist = fit_to(corpus, smoothing=False)
        program = converter.to_type_annotated_ns_s_exp(
            ns.python_to_python_ast(program), dfa, "M"
        )
        like = fam.compute_likelihood(dist, program)
        like = Fraction.from_float(float(np.exp(like))).limit_denominator()
        results = fam.compute_likelihood_per_node(dist, program)
        results = [
            (
                ns.render_s_expression(x),
                Fraction.from_float(float(np.exp(y))).limit_denominator(),
            )
            for x, y in results
            if y != 0  # remove zero log-likelihoods
        ]
        print(like)
        print(results)
        return like, results

    def test_likelihood(self):
        like, results = self.compute_likelihood(["x = 2", "y = 3", "y = 4"], "y = 4")
        self.assertAlmostEqual(like, Fraction(2, 9))
        self.assertEqual(
            results,
            [
                ("(const-&y:0~Name)", Fraction(2, 3)),
                ("(const-i4~Const)", Fraction(1, 3)),
            ],
        )

    def test_likelihood_def_use_check(self):
        like, results = self.compute_likelihood(
            ["x = 2; y = x", "y = 2; x = y"], "x = 2; y = x"
        )
        self.assertAlmostEqual(like, Fraction(1, 8))
        self.assertEqual(
            results,
            [
                ("(const-&x:0~Name)", Fraction(1, 2)),
                ("(const-&y:0~Name)", Fraction(1, 2)),
                ("(Name~E (const-&x:0~Name) (Load~Ctx))", Fraction(1, 2)),
            ],
        )

    def test_likelihood_zero(self):
        like, results = self.compute_likelihood(
            ["y = x + 2", "y = 2 + 3", "y = 4"], "y = 2 + x"
        )
        self.assertAlmostEqual(like, Fraction(0))
        self.assertEqual(
            results,
            [
                (
                    "(BinOp~E (Constant~E (const-i2~Const) (const-None~ConstKind)) (Add~O) (Name~E (const-g_x~Name) (Load~Ctx)))",
                    Fraction(2, 3),
                ),
                (
                    "(Constant~E (const-i2~Const) (const-None~ConstKind))",
                    Fraction(1, 2),
                ),
                ("(const-i2~Const)", Fraction(1, 2)),
                ("(Name~E (const-g_x~Name) (Load~Ctx))", Fraction(0, 1)),
            ],
        )

    def test_likelihood_with_abstractions(self):
        # test from annie
        # I don't think it actually makes sense since (fn_3) shouldn't be possible
        test_programs = ["(fn_1 (fn_2) (fn_2))", "(fn_1 (fn_3 (fn_3)) (fn_3))"]
        test_programs_ast = [converter.s_exp_to_python_ast(p) for p in test_programs]
        test_dfa = {"E": {"fn_1": ["E", "E"], "fn_2": [], "fn_3": ["E"]}}

        test_subset = DSLSubset.from_program(
            test_dfa,
            *test_programs_ast,
            root="E",
        )
        test_dsl = create_dsl(test_dfa, test_subset, "E")

        test_fam = ns.BigramProgramDistributionFamily(test_dsl)
        test_counts = test_fam.count_programs(
            [
                [
                    converter.to_type_annotated_ns_s_exp(
                        test_programs_ast[0], test_dfa, "E"
                    )
                ]
            ]
        )
        test_dist = test_fam.counts_to_distribution(test_counts)[0]
        likelihood = test_fam.compute_likelihood(
            test_dist,
            converter.to_type_annotated_ns_s_exp(test_programs_ast[1], test_dfa, "E"),
        )
        self.assertEqual(likelihood, -np.inf)
        result = test_fam.compute_likelihood_per_node(
            test_dist,
            converter.to_type_annotated_ns_s_exp(test_programs_ast[1], test_dfa, "E"),
        )
        result = [
            (
                ns.render_s_expression(x),
                Fraction.from_float(float(np.exp(y))).limit_denominator(),
            )
            for x, y in result
            if y != 0  # remove zero log-likelihoods
        ]
        self.assertEqual(
            result,
            [
                ("(fn_3~E (fn_3~E))", Fraction(0, 1)),
                ("(fn_3~E)", Fraction(0, 1)),
                ("(fn_3~E)", Fraction(0, 1)),
            ],
        )