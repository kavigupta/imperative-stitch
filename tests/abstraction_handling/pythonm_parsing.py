import unittest

from imperative_stitch.compress.pythonm import parse_pythonm

from .abstraction_test import assertSameCode, fn_2, fn_2_args


class TestPythonMParsing(unittest.TestCase):

    def test_stub_parsing(self):
        stub = fn_2.create_stub(fn_2_args, is_pythonm=True)
        assertSameCode(
            self,
            stub.to_python(),
            """
            fn_2(`print(2)`, &c, &a, &b, &d, `if x == 3:\\n    pass`)
            """,
        )
        self.assertEqual(stub, parse_pythonm(stub.to_python()))
