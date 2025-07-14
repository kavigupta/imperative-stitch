import ast
import copy


def remove_docstrings(code: ast.AST) -> ast.AST:
    """
    Remove docstrings from the code. Does not modify the input code.
    """
    code = copy.deepcopy(code)
    for node in ast.walk(code):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if is_docstring(node.body[0]):
                node.body = node.body[1:]
                if not node.body:
                    node.body.append(ast.Pass())
    return code


def is_docstring(node: ast.AST) -> bool:
    """
    Check if the node is a docstring.
    """
    if not isinstance(node, ast.Expr):
        return False
    if not isinstance(node.value, ast.Constant):
        return False
    if not isinstance(node.value.value, str):
        return False
    return True
