import ast


def replace_break_and_continue(function_def, replace_with):
    """
    Replace top-level break and continue statements with the given statement.
    Do *not* replace break and continue statements inside loops or inner functions.

    Args:
        function_def (ast.FunctionDef): The function definition to modify.
        replace_with (ast.AST): The AST node to replace break and continue statements with.

    Returns:
        ast.FunctionDef: The modified function definition.
    """
    assert isinstance(function_def, ast.FunctionDef), function_def
    transformer = ReplaceBreakAndContinue(replace_with)
    function_def.body = [transformer.visit(x) for x in function_def.body]
    return transformer.has_break, transformer.has_continue, function_def


class ReplaceBreakAndContinue(ast.NodeTransformer):
    """Replace break and continue statements with the given statement.

    Attributes:
        has_break (bool): True if the function contained a break statement.
        has_continue (bool): True if the function contained a continue statement.
    """

    def __init__(self, replace_with):
        super().__init__()
        self.replace_with = replace_with
        self.has_break = False
        self.has_continue = False

    def visit_Break(self, node):
        self.has_break = True
        return self.replace_with

    def visit_Continue(self, node):
        self.has_continue = True
        return self.replace_with

    def visit_FunctionDef(self, node):
        return node

    def visit_While(self, node):
        return node

    def visit_For(self, node):
        return node