import ast
import tokenize
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from types import NoneType
from typing import Dict, List, Tuple, Union

import neurosym as ns

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import collect_abstraction_calls
from imperative_stitch.parser.python_ast import AbstractionCallAST


def parse_with_target_state(code: str, target_state: str) -> ns.PythonAST:
    if target_state == "M":
        return ns.python_to_python_ast(code)
    if target_state == "seqS":
        return ns.python_statements_to_python_ast(code)
    if target_state == "S":
        return ns.python_statement_to_python_ast(code)
    if target_state == "E":
        stmt = parse_with_target_state(code, "S")
        assert isinstance(stmt, ns.NodeAST) and stmt.typ == ast.Expr
        return stmt.children[0]
    raise ValueError(f"Unsupported target state: {target_state}")


def parse_pythonm(
    code: str, target_state: str, abstractions: Dict[str, Abstraction]
) -> ns.PythonAST:
    """
    Parses a PythonM code string into a neurosym PythonAST object.

    See manipulate_python_ast.py for details on how PythonM code is structured, but
    basically there's two additions: &name for symvars and `code` for codevars.
    """
    code = replace_pythonm_with_normal_stub(code)
    code = parse_with_target_state(code, target_state)
    return convert_all_calls_to_abstractions(code, abstractions)


def convert_all_calls_to_abstractions(code, abstractions):
    code = code.map(lambda node: function_call_to_abstraction_call(node, abstractions))
    existing_calls = collect_abstraction_calls(code)
    if not existing_calls:
        return code

    get_root_state = lambda node: abstractions[node.tag].dfa_root

    def manipulate_call_to_correct_symbol(node):
        if isinstance(node, AbstractionCallAST):
            if get_root_state(node) == "E":
                del existing_calls[node.handle]
            for arg, state in zip(
                node.args, abstractions[node.tag].all_argument_states
            ):
                if not isinstance(arg, AbstractionCallAST):
                    continue
                assert state == get_root_state(arg)
                if arg.handle in existing_calls:
                    del existing_calls[arg.handle]
            return node
        if isinstance(node, ns.NodeAST) and node.typ == ast.Expr:
            expr = node.children[0]
            if not isinstance(expr, AbstractionCallAST):
                return node
            state = get_root_state(expr)
            if state not in {"seqS", "S"}:
                return node
            node = expr
            if state == "seqS":
                node = ns.SpliceAST(node)
            del existing_calls[expr.handle]
            return node
        if isinstance(node, ns.SpliceAST):
            if isinstance(node.content, AbstractionCallAST):
                assert get_root_state(node.content) == "seqS"
                del existing_calls[node.content.handle]
        return node

    result = code.map(manipulate_call_to_correct_symbol)
    if not existing_calls:
        return result
    raise ValueError(
        f"Failed to convert all calls to abstractions. Remaining calls: {existing_calls}; abstractions: {abstractions}"
    )


def parse_function_call(
    node: ns.PythonAST, *, validate
) -> Union[Tuple[str, List[ns.PythonAST]], NoneType]:
    """
    Parses a function call node into its function name and arguments. It also checks that
    there are no kwargs. If the node is not a valid function call, it returns None. If
    validate_kwargs is True, it raises an error if there are any keyword arguments, otherwise
    it returns None
    """
    if not isinstance(node, ns.NodeAST) or node.typ != ast.Call:
        return None
    func, args, kwargs = node.children
    if not isinstance(func, ns.NodeAST) or func.typ != ast.Name:
        return None
    func_name = func.children[0]
    assert isinstance(args, ns.ListAST) and isinstance(kwargs, ns.ListAST)
    if len(kwargs.children) > 0:
        if validate:
            raise ValueError("PythonM calls should not have keyword arguments")
        return None
    args = args.children
    if not all(isinstance(arg, ns.StarrableElementAST) for arg in args):
        if validate:
            raise ValueError(
                "PythonM calls should only have StarrableElementAST arguments"
            )
        return None
    args = [arg.content for arg in args]
    return func_name.leaf.name, args


def function_call_to_abstraction_call(
    call_node, abstractions: Dict[str, Abstraction]
) -> ns.PythonAST:
    attempt_parse = parse_function_call(call_node, validate=False)
    if attempt_parse is None:
        return call_node
    func_name, args = attempt_parse

    if not func_name.startswith("fn_"):
        return call_node

    args = [
        extract_pythonm_argument(arg, state, abstractions)
        for arg, state in zip(args, abstractions[func_name].all_argument_states)
    ]

    return AbstractionCallAST(tag=func_name, args=args, handle=uuid.uuid4())


def extract_pythonm_argument(
    arg: ns.PythonAST, state: str, abstractions: Dict[str, Abstraction]
) -> ns.PythonAST:
    parsed = parse_function_call(arg, validate=True)
    assert parsed is not None, f"Invalid PythonM argument: {arg}"
    func_name, args = parsed
    assert len(args) == 1, f"PythonM argument should have exactly one argument: {args}"
    arg = args[0]
    if func_name == "__code__":
        assert isinstance(arg, ns.NodeAST) and arg.typ == ast.Constant
        arg = arg.children[0].leaf
        return parse_pythonm(arg, state, abstractions)
    if func_name == "__ref__":
        assert isinstance(arg, ns.NodeAST) and arg.typ == ast.Name
        arg = arg.children[0].leaf
        return ns.LeafAST(ns.PythonSymbol(arg.name, scope=None))
    raise ValueError(f"Unknown PythonM function: {func_name} in {arg}")


def is_backtick(tok: tokenize.TokenInfo) -> bool:
    """
    Check if the token is a backtick.
    """
    return tok.type in {tokenize.OP, tokenize.ERRORTOKEN} and tok.string == "`"


@dataclass
class PythonMStateMachine:
    tokens: List[tokenize.TokenInfo]
    backtick_depth: int = 0
    open_locs: List[int] = field(default_factory=list)
    code_blocks: List[Tuple[int, int]] = field(default_factory=list)
    string_replacements: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_code(cls, code: str):
        tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
        return cls(tokens)

    def process_token(self, i):
        tok = self.tokens[i]
        if is_backtick(tok):
            if starting_context(self.tokens, i - 1):
                self.process_opening_backick(i)
            else:
                self.process_closing_backick(i)
        if tok.type == tokenize.OP and tok.string == "&":
            if starting_context(self.tokens, i - 1):
                self.string_replacements[i] = "__ref__("
                self.string_replacements[i + 1] = self.tokens[i + 1].string + ")"

    def process_opening_backick(self, i):
        if self.backtick_depth == 0:
            self.open_locs.append(i)
        self.backtick_depth += 1
        self.string_replacements[i] = "__code__("

    def process_closing_backick(self, i):
        assert self.backtick_depth > 0, f"Unexpected closing backtick at {i}"
        self.backtick_depth -= 1
        self.string_replacements[i] = ")"
        if self.backtick_depth != 0:
            # If we are still inside backticks, just return. We will handle this in a recursive case.
            return
        corresponding_open = self.open_locs.pop()
        if corresponding_open == i - 1:
            # special case of ``. We handle this here rather than in the code blocks.
            self.string_replacements[i] = repr("") + ")"
        else:
            # we use code blocks here because it allows much easier calling of the locations
            # since we don't actually have indices into the original string at this point.
            for j in range(corresponding_open + 1, i):
                if j in self.string_replacements:
                    del self.string_replacements[j]
            self.code_blocks.append((corresponding_open + 1, i - 1))

    def process(self):
        """
        Process the tokens and replace backticks and & with the appropriate PythonM syntax.
        """
        for i in range(len(self.tokens)):
            self.process_token(i)

        assert (
            self.backtick_depth == 0
        ), f"Unclosed backticks at the end: {self.backtick_depth}"
        return self.tokens, self.string_replacements, self.code_blocks


def replace_pythonm_with_normal_stub(code):
    return perform_replacements(code, *PythonMStateMachine.from_code(code).process())


def perform_replacements(code, tokens, replacements, code_blocks):
    """
    Perform the replacements in the code based on the tokens and replacements mapping.

    Code blocks are inclusive in token space.
    """
    code_as_lines = code.split("\n")
    line_offsets = [0]
    for line in code_as_lines:
        line_offsets.append(
            line_offsets[-1] + len(line) + 1
        )  # +1 for the newline character

    def get_token_at(i):
        tok = tokens[i]
        line_no = tok.start[0] - 1
        start, end = tok.start[1], tok.end[1]
        start = line_offsets[line_no] + start
        end = line_offsets[line_no] + end
        assert code[start:end] == tok.string  # sanity check
        return start, end

    replacements_by_range = []
    for i, replacement in replacements.items():
        start, end = get_token_at(i)
        replacements_by_range.append((start, end, replacement))
    for start_tok, end_tok in code_blocks:
        assert start_tok <= end_tok
        start, end = get_token_at(start_tok)[0], get_token_at(end_tok)[1]
        replacements_by_range.append((start, end, repr(code[start:end])))

    characters = list(code)
    for start, end, replacement in replacements_by_range:
        assert start < end
        characters[start] = replacement
        for i in range(start + 1, end):
            characters[i] = ""
    code = "".join(characters)
    return code


def starting_context(tokens, loc):
    # skip whitespace
    while loc >= 0 and tokens[loc].string.strip() == "":
        loc -= 1
    tok = tokens[loc]
    # either ( or ,
    return tok.type == tokenize.OP and tok.string in ("(", ",")
