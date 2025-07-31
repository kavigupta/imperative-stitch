import ast
import unittest
from textwrap import dedent

import neurosym as ns
import numpy as np

from imperative_stitch.parser import converter
from imperative_stitch.utils.classify_nodes import export_dfa
from tests.dsl_tests.utils import fit_to


from .utils import expand_with_slow_tests, small_set_examples


class Cleanup(ast.NodeTransformer):
    # drop the kind field from Constant nodes
    def visit_Constant(self, node):
        return ast.Constant(node.value)


def cleanup_code(code):
    return ast.unparse(Cleanup().visit(ast.parse(code)))


def all_variables_global(program):
    # replace e.g., const-&x:0 with const-g_x
    import re

    return re.sub(r"const-&(\w+):\d+", r"const-g_\1", program)


class CheckSamplesValid(unittest.TestCase):
    def check_samples(self, *programs, count=100):
        programs = [cleanup_code(p) for p in programs]
        dfa = export_dfa()
        _, _, fam, dist = fit_to(programs)
        samples = [fam.sample(dist, np.random.RandomState(0)) for _ in range(count)]
        samples = [ns.render_s_expression(x) for x in samples]
        for samp in samples:
            py_code = converter.s_exp_to_python(samp)
            print(samp)
            samp_reconstructed = converter.to_type_annotated_ns_s_exp(
                ns.python_to_python_ast(py_code), dfa, "M"
            )
            samp_reconstructed = ns.render_s_expression(samp_reconstructed)
            print(py_code)
            samp_to_check = all_variables_global(samp)
            samp_reconstructed = all_variables_global(samp_reconstructed)
            print("*" * 80)
            print(samp_to_check)
            print(samp_reconstructed)
            self.maxDiff = None
            self.assertEqual(samp_to_check, samp_reconstructed)
        return samples

    def test_check_samples_basic(self):
        self.check_samples("x = y + 2 + 2")

    def test_unicode_string(self):
        self.check_samples("x, y = u'a', 2")

    def test_local_global(self):
        self.check_samples(
            dedent(
                """
                if x == 2:
                    x = 2
                else:
                    print(x)
                """
            )
        )

    def test_alternate_lengths_of_keywords(self):
        self.check_samples(
            dedent(
                """
                def f(x=2, y=3): pass
                def g(x=2, y=3, z=4): pass
                """
            ),
        )

    def test_temp(self):
        self.check_samples(
            r"""
            def f(x=2, y=3): pass
            def g(x=2, y=3, z=4): pass
            """,
            count=1000,
        )

    @expand_with_slow_tests(len(small_set_examples()))
    def test_realistic(self, i):
        self.check_samples(small_set_examples()[i])
