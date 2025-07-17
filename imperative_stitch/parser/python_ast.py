import ast
import uuid
from dataclasses import dataclass
from typing import List

import neurosym as ns
from frozendict import frozendict


@dataclass
class Variable(ns.PythonAST):
    sym: str

    @property
    def idx(self):
        return int(self.sym[1:])

    def map(self, fn):
        return fn(self)

    def to_ns_s_exp(self, config=frozendict()):
        if config.get("no_leaves", False):
            return ns.SExpression("var-" + self.sym, [])
        return self.sym

    def is_multiline(self):
        return False


@dataclass
class SymvarAST(Variable):
    def to_python_ast(self):
        return self.sym

    def _replace_with_substitute(self, arguments):
        return arguments.symvars[self.idx - 1]


@dataclass
class MetavarAST(Variable):
    def to_python_ast(self):
        return ast.Name(id=self.sym)

    def _replace_with_substitute(self, arguments):
        return arguments.metavars[self.idx]


@dataclass
class ChoicevarAST(Variable):
    def to_python_ast(self):
        return ast.Name(id=self.sym)

    def _replace_with_substitute(self, arguments):
        return ns.SpliceAST(arguments.choicevars[self.idx])


@dataclass
class AbstractionCallAST(ns.PythonAST):
    tag: str
    args: List[ns.PythonAST]
    handle: uuid.UUID

    def to_ns_s_exp(self, config=frozendict()):
        return ns.SExpression(self.tag, [x.to_ns_s_exp(config) for x in self.args])

    def to_python_ast(self):
        raise RuntimeError("cannot convert abstraction call to python")

    def map(self, fn):
        return fn(
            AbstractionCallAST(self.tag, [x.map(fn) for x in self.args], self.handle)
        )

    def _replace_with_substitute(self, arguments):
        del arguments  # no need for arguments in this case, since this is not a variable
        # all we are doing is replacing the handle with a new one, so that we can
        # distinguish between different calls to the same abstraction
        return type(self)(self.tag, self.args, uuid.uuid4())

    def is_multiline(self):
        # In any context where this is used, it is not multiline.
        # This is because it is a call to an abstraction, which is always a single line
        # Any code is going to be made single line in the process of serialization.
        return False
