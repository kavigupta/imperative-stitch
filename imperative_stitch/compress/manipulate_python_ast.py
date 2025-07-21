import ast
from dataclasses import dataclass

import neurosym as ns
from frozendict import frozendict


@dataclass
class AddressOfSymbolAST(ns.PythonAST):
    sym: str

    def to_python_ast(self):
        # ctx does not matter here, so just use Load
        return ast.Name(id="&" + self.sym, ctx=ast.Load())

    def map(self, fn):
        return AddressOfSymbolAST(fn(self.sym))

    def to_ns_s_exp(self, config=frozendict()):
        return "&" + self.sym

    def is_multiline(self):
        return False


def render_symvar(node, *, is_pythonm):
    """
    Render this PythonAST as a __ref__ variable for stub display, i.e.,
        `a` -> `__ref__(a)`
    """
    if is_pythonm:
        return AddressOfSymbolAST(node.leaf.name)
    return ns.make_python_ast.make_call(
        ns.PythonSymbol(name="__ref__", scope=None), ns.make_python_ast.make_name(node)
    )


@dataclass
class QuotedCodeAST(ns.PythonAST):
    content: ns.PythonAST

    def to_python_ast(self):
        """
        Convert this QuotedCodeAST to a Python code object.
        """
        code_content = self.content.to_python()
        assert "\n" not in code_content, "QuotedCodeAST cannot contain newlines"
        return ast.Name(
            id=f"`{code_content}`",
            # this doesn't actually matter, so just use a dummy context
            ctx=ast.Load(),
        )

    def map(self, fn):
        return QuotedCodeAST(fn(self.content.map(fn)))

    def to_ns_s_exp(self, config=frozendict()):
        return self.content.to_ns_s_exp(config=config)

    def is_multiline(self):
        return self.content.is_multiline()


def render_codevar(node, *, is_pythonm):
    """
    Render this PythonAST as a __code__ variable for stub display, i.e.,
        `a` -> `__code__("a")`
    """
    if is_pythonm:
        return QuotedCodeAST(node)
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
