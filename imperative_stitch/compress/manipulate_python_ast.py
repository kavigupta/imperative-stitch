import ast

import neurosym as ns


def render_symvar(node, *, is_pythonm):
    """
    Render this PythonAST as a __ref__ variable for stub display, i.e.,
        `a` -> `__ref__(a)`
    """
    if is_pythonm:
        return ns.make_python_ast.make_name("&" + node)
    return ns.make_python_ast.make_call(
        ns.PythonSymbol(name="__ref__", scope=None), ns.make_python_ast.make_name(node)
    )


# class QuotedCodeAST(PythonAST):
#     pass


def render_codevar(node, *, is_pythonm):
    """
    Render this PythonAST as a __code__ variable for stub display, i.e.,
        `a` -> `__code__("a")`
    """
    if is_pythonm:
        return ns.make_python_ast.make_name("`" + node + "`")
    return ns.make_python_ast.make_call(
        ns.PythonSymbol(name="__code__", scope=None),
        ns.make_python_ast.make_constant(node.to_python()),
    )


def wrap_in_metavariable(node, name):
    return ns.NodeAST(
        ast.Set,
        [
            ns.ListAST(
                [
                    ns.make_python_ast.make_name(
                        ns.LeafAST(ns.PythonSymbol("__metavariable__", None))
                    ),
                    ns.make_python_ast.make_name(
                        ns.LeafAST(ns.PythonSymbol(name, None))
                    ),
                    node,
                ]
            )
        ],
    )


def wrap_in_choicevar(node):
    return ns.SequenceAST(
        "/seq",
        [
            ns.python_statement_to_python_ast("__start_choice__"),
            node,
            ns.python_statement_to_python_ast("__end_choice__"),
        ],
    )
