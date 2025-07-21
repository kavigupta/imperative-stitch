import unittest

import neurosym as ns

from imperative_stitch.compress.pythonm import parse_pythonm

from .abstraction_test import assertSameCode, fn_2, fn_2_args_w_nothing


class TestPythonMParsing(unittest.TestCase):

    def test_stub_parsing(self):
        stub = fn_2.create_stub(
            fn_2_args_w_nothing[:-1]
            + [ns.python_statements_to_python_ast("x = 2 if 3 else 9")],
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
        self.assertEqual(
            ns.render_s_expression(stub.to_ns_s_exp()),
            ns.render_s_expression(parse_pythonm(stub.to_python()).to_ns_s_exp()),
        )
