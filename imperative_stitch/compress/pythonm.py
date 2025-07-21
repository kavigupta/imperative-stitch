import ast
from io import BytesIO
import tokenize
from types import NoneType
from typing import Dict, List, Tuple, Union
import uuid
import neurosym as ns

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_python_ast import AddressOfSymbolAST
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
    print(code)
    code = replace_pythonm_with_normal_stub(code)
    print(code)
    code = parse_with_target_state(code, target_state)
    print(code)
    return code.map(lambda node: function_call_to_abstraction_call(node, abstractions))


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
    elif func_name == "__ref__":
        print(arg)
        assert isinstance(arg, ns.NodeAST) and arg.typ == ast.Name
        arg = arg.children[0].leaf
        return ns.LeafAST(ns.PythonSymbol(arg.name, scope=None))
    else:
        raise ValueError(f"Unknown PythonM function: {func_name} in {arg}")


def replace_pythonm_with_normal_stub(code):
    tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
    backtick_depth = 0
    last_open = None
    code_blocks = []
    string_replacements = {}
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.OP and tok.string == "`":
            if starting_context(tokens[i - 1]):
                if backtick_depth == 0:
                    assert (
                        last_open is None
                    ), f"Unexpected backtick without closing at {i}"
                    last_open = i
                backtick_depth += 1
                string_replacements[i] = "__code__("
            else:
                backtick_depth -= 1
                string_replacements[i] = ")"
                if backtick_depth == 0:
                    assert last_open is not None, f"Unexpected closing backtick at {i}"
                    for j in range(last_open + 1, i):
                        if j in string_replacements:
                            del string_replacements[j]
                    code_blocks.append((last_open + 1, i - 1))
                    last_open = None
        if tok.type == tokenize.OP and tok.string == "&":
            assert starting_context(tokens[i - 1])
            string_replacements[i] = "__ref__("
            string_replacements[i + 1] = tokens[i + 1].string + ")"
    return perform_replacements(code, tokens, string_replacements, code_blocks)


def perform_replacements(code, tokens, replacements, code_blocks):
    """
    Perform the replacements in the code based on the tokens and replacements mapping.
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
        start, end = get_token_at(start_tok)[0], get_token_at(end_tok)[1]
        replacements_by_range.append((start, end, repr(code[start:end])))
    print(tokens)
    print(replacements_by_range)

    characters = list(code)
    for start, end, replacement in replacements_by_range:
        characters[start] = replacement
        for i in range(start + 1, end):
            characters[i] = ""
    code = "".join(characters)
    return code


def starting_context(tok):
    # either ( or ,
    return tok.type == tokenize.OP and tok.string in ("(", ",")


def ending_context(tok):
    # either ) or ,
    return tok.type == tokenize.OP and tok.string in (")", ",")
