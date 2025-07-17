from io import BytesIO
import tokenize
import neurosym as ns


def parse_pythonm(code: str) -> ns.PythonAST:
    """
    Parses a PythonM code string into a neurosym PythonAST object.

    See manipulate_python_ast.py for details on how PythonM code is structured, but
    basically there's two additions: &name for symvars and `code` for codevars.
    """
    code = "fn_2(`print(2)`, &c, &a, &b, &d, `if x == 3:\\n    pass`)"
    print(repr(code))
    tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
    print(tokens)
    print([t.string for t in tokens])
    1 / 0
