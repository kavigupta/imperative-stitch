from dataclasses import dataclass
from typing import List

import neurosym as ns

from imperative_stitch.compress.manipulate_python_ast import (
    render_codevar,
    render_symvar,
    wrap_in_choicevar,
    wrap_in_metavariable,
)
from imperative_stitch.parser import converter
from imperative_stitch.parser.patterns import VARIABLE_PATTERN
from imperative_stitch.parser.python_ast import AbstractionCallAST, Variable
from imperative_stitch.utils.classify_nodes import export_dfa


@dataclass
class Arguments:
    """
    Represents the arguments to an abstraction function. This is a list of metavariables,
        symbol variables, and choice variables.
    """

    metavars: list[ns.PythonAST]
    symvars: list[ns.PythonAST]
    choicevars: list[ns.SequenceAST]

    def __post_init__(self):
        assert all(isinstance(x, ns.PythonAST) for x in self.metavars), self.metavars
        assert all(isinstance(x, ns.PythonAST) for x in self.symvars), self.symvars
        assert all(
            isinstance(x, (ns.SequenceAST, Variable, AbstractionCallAST))
            for x in self.choicevars
        ), self.choicevars

    @classmethod
    def from_list(cls, arguments, arity, sym_arity, choice_arity):
        assert len(arguments) == arity + sym_arity + choice_arity
        metavars, symvars, choicevars = (
            arguments[:arity],
            arguments[arity : arity + sym_arity],
            arguments[arity + sym_arity :],
        )
        return cls(metavars, symvars, choicevars)

    def render_list(self, *, is_pythonm):
        return (
            [render_codevar(x, is_pythonm=is_pythonm) for x in self.metavars]
            + [render_symvar(x, is_pythonm=is_pythonm) for x in self.symvars]
            + [render_codevar(x, is_pythonm=is_pythonm) for x in self.choicevars]
        )


@dataclass
class Abstraction:
    name: str
    body: ns.PythonAST
    arity: int

    sym_arity: int
    choice_arity: int

    dfa_root: str
    dfa_symvars: list[str]
    dfa_metavars: list[str]
    dfa_choicevars: list[str]

    @classmethod
    def of(
        cls,
        name,
        body,
        dfa_root,
        *,
        dfa_symvars=(),
        dfa_metavars=(),
        dfa_choicevars=(),
        arity=None,
        sym_arity=None,
        choice_arity=None,
    ):
        if isinstance(body, str):
            body = converter.s_exp_to_python_ast(body)
        if arity is not None:
            assert arity == len(dfa_metavars)
        if sym_arity is not None:
            assert sym_arity == len(dfa_symvars)
        if choice_arity is not None:
            assert choice_arity == len(dfa_choicevars)
        dfa_symvars = list(dfa_symvars)
        dfa_metavars = list(dfa_metavars)
        dfa_choicevars = list(dfa_choicevars)
        return cls(
            name=name,
            body=body,
            arity=len(dfa_metavars),
            sym_arity=len(dfa_symvars),
            choice_arity=len(dfa_choicevars),
            dfa_root=dfa_root,
            dfa_symvars=dfa_symvars,
            dfa_metavars=dfa_metavars,
            dfa_choicevars=dfa_choicevars,
        )

    def __post_init__(self):
        assert self.arity == len(self.dfa_metavars)
        assert self.sym_arity == len(self.dfa_symvars)
        assert self.choice_arity == len(self.dfa_choicevars)

        assert isinstance(self.body, ns.PythonAST), self.body

    def process_arguments(self, arguments):
        """
        Produce an Arguments object from a list of arguments.
        """
        return Arguments.from_list(
            arguments, self.arity, self.sym_arity, self.choice_arity
        )

    def create_stub(self, arguments, *, is_pythonm):
        """
        Create a stub of this abstraction with the given arguments. The stub looks something
            like `fn_1(__code__("a + b"), __ref__(a), __ref__(z), __code("x = 2 + 3"))`,
            where the metavariables and choice variables are __code__, and the symbol
            variables are __ref__.
        """
        arguments = self.process_arguments(arguments)
        args_list = arguments.render_list(is_pythonm=is_pythonm)
        e_stub = ns.make_python_ast.make_call(
            ns.PythonSymbol(name=self.name, scope=None),
            *args_list,
        )
        if self.dfa_root == "E":
            return e_stub
        s_stub = ns.make_python_ast.make_expr_stmt(e_stub)
        if self.dfa_root == "S":
            return s_stub
        seq_stub = ns.SequenceAST("/seq", [s_stub])
        assert self.dfa_root == "seqS"
        return seq_stub

    def _add_extract_pragmas(self, body):
        if self.dfa_root == "E":
            raise ValueError("Cannot add extract pragmas to an expression")
        start_pragma = ns.python_statement_to_python_ast("__start_extract__")
        end_pragma = ns.python_statement_to_python_ast("__end_extract__")
        if self.dfa_root == "S":
            return ns.SpliceAST(
                ns.SequenceAST("/seq", [start_pragma, body, end_pragma])
            )
        assert self.dfa_root == "seqS"
        return ns.SequenceAST("/seq", [start_pragma, *body.elements, end_pragma])

    def substitute_body(self, arguments, *, pragmas=False):
        """
        Substitute the given arguments into the body of this abstraction.
        """
        if not isinstance(arguments, Arguments):
            arguments = self.process_arguments(arguments)
        if pragmas:
            if not all(x == "E" for x in self.dfa_metavars):
                raise ValueError(
                    "Cannot add extract pragmas to a body with non-expression metavariables"
                )
            if not all(x == "seqS" for x in self.dfa_choicevars):
                raise ValueError(
                    "Cannot add extract pragmas to a body with non-statement choicevars"
                )
            arguments = Arguments(
                [
                    wrap_in_metavariable(x, f"__m{i}")
                    for i, x in enumerate(arguments.metavars)
                ],
                arguments.symvars,
                [wrap_in_choicevar(x) for x in arguments.choicevars],
            )
        body = self.body
        body = body.map(
            lambda x: (
                # pylint: disable=protected-access
                x._replace_with_substitute(arguments)
                if hasattr(x, "_replace_with_substitute")
                else x
            )
        )
        if pragmas:
            body = self._add_extract_pragmas(body)
        return body

    def body_with_variable_names(self, *, pragmas=False):
        """
        Render the body but with the #0, %0, ?0, kept as placeholders.
        """
        arguments = Arguments(
            [
                ns.make_python_ast.make_name(ns.LeafAST(ns.PythonSymbol(f"#{i}", None)))
                for i in range(self.arity)
            ],
            [
                ns.LeafAST(ns.PythonSymbol(f"%{i + 1}", None))
                for i in range(self.sym_arity)
            ],
            [
                ns.SequenceAST(
                    "/seq",
                    [
                        ns.make_python_ast.make_expr_stmt(
                            ns.make_python_ast.make_name(
                                ns.LeafAST(ns.PythonSymbol(f"?{i}", None))
                            )
                        )
                    ],
                )
                for i in range(self.choice_arity)
            ],
        )
        return self.substitute_body(arguments, pragmas=pragmas)

    def variables_in_order(self, node_ordering, previous_abstractions=()) -> List[str]:
        """
        Return a list of all the metavariables, symbol variables, and choice variables in
            the order they appear in the body.
        """
        result = []
        seen = set()

        def traverse(node):
            sym = node.symbol
            sym_mat = VARIABLE_PATTERN.match(sym)
            if sym_mat:
                var = sym_mat.group("name")
                if var not in seen:
                    seen.add(var)
                    result.append(var)
            ordering = (
                node_ordering[node.symbol]
                if sym in node_ordering
                else range(len(node.children))
            )
            for i in ordering:
                traverse(node.children[i])

        body = ns.to_type_annotated_ns_s_exp(
            self.body, export_dfa(abstrs=previous_abstractions), self.dfa_root
        )
        traverse(body)
        return result

    def arguments_traversal_order(self, node_ordering, previous_abstractions=()):
        """
        Return a list of indices that can be used to traverse the arguments in the order
            they appear in the body.
        """
        arguments = []
        arguments += [f"#{i}" for i in range(self.arity)]
        arguments += [f"%{i + 1}" for i in range(self.sym_arity)]
        arguments += [f"?{i}" for i in range(self.choice_arity)]
        arguments = {x: i for i, x in enumerate(arguments)}
        vars_in_order = self.variables_in_order(node_ordering, previous_abstractions)
        return [arguments[x] for x in vars_in_order]

    def __repr__(self):
        this_as_dict = {}
        for k, v in self.__dict__.items():
            if k != "body":
                this_as_dict[k] = repr(v)
                continue
            v = ns.render_s_expression(v.to_ns_s_exp())
            this_as_dict[k] = f"converter.s_exp_to_python_ast({v!r})"
        return (
            "Abstraction("
            + ", ".join(f"{k}={v}" for k, v in this_as_dict.items())
            + ")"
        )


def handle_abstractions(name_to_abstr):
    """
    Given a dictionary mapping abstraction names to Abstraction objects, return a function
        that can be used to inject abstractions into a PythonAST.
    """

    def inject(abstr_name, abstr_args, pair_to_s_exp):
        abst = name_to_abstr[abstr_name]
        o = pair_to_s_exp(abst.body_s_expr(abstr_args))
        return o

    return inject
