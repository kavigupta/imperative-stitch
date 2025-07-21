import unittest

import neurosym as ns

from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies,
)
from imperative_stitch.compress.pythonm import parse_pythonm

from .abstraction_test import assertSameCode, fn_2, fn_2_args_w_nothing


class TestPythonMParsing(unittest.TestCase):

    def test_stub_parsing(self):
        abstractions = {"fn_2": fn_2}
        args = fn_2_args_w_nothing[:-1] + [
            ns.python_statements_to_python_ast("x = 2 if 3 else 9")
        ]
        stub = fn_2.create_stub(
            args,
            is_pythonm=True,
        )
        print(stub.to_ns_s_exp())
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(`print(2)`, &c, &a, &b, &d, `x = 2 if 3 else 9`)
            """,
        )
        stub_back_forth = parse_pythonm(stub.to_python(), fn_2.dfa_root, abstractions)

        print(ns.render_s_expression(stub.to_ns_s_exp()))
        print(ns.render_s_expression(stub_back_forth.to_ns_s_exp()))
        print("X" * 100)

        original_inlined = fn_2.substitute_body(args)
        back_fourth_inlined = abstraction_calls_to_bodies(
            stub_back_forth, abstractions
        )

        print(ns.render_s_expression(original_inlined.to_ns_s_exp()))
        print(ns.render_s_expression(back_fourth_inlined.to_ns_s_exp()))

        print(back_fourth_inlined)

        self.maxDiff = 1000

        self.assertEqual(
            original_inlined.to_python(),
            back_fourth_inlined.to_python(),
        )
