import ast

from ast_scope.group_similar_constructs import GroupSimilarConstructsVisitor
from ast_scope.scope import FunctionScope


class NodeToContainingFunction(GroupSimilarConstructsVisitor):
    """
    Computes the containing functions of each node in the AST.

    Includes function definitions, lambdas, and comprehensions.
    """

    def __init__(self):
        self.current_fn = []
        self.node_to_containing = {}

    def generic_visit(self, node):
        if node is None:
            return
        if isinstance(node, list):
            for x in node:
                self.visit(x)
            return
        super().generic_visit(node)
        self.node_to_containing[node] = self.current_fn

    def visit_defaults_of(self, args):
        self.visit(args.defaults + args.kw_defaults)

    def visit_function_def(self, func_node, is_async):
        self.visit_defaults_of(func_node.args)
        self.visit(func_node.decorator_list)
        self.visit_scope(func_node, func_node.body)

    def visit_comprehension_generic(self, targets, comprehensions, node):
        self.visit(self.get_generators(node))
        return self.visit_scope(node, [node.elt, self.get_ifs(node)])

    # The argument name changed from `comp_node` to `node` in later versions of Python.
    # pylint: disable=arguments-renamed
    def visit_DictComp(self, node):
        self.visit(self.get_generators(node))
        return self.visit_scope(
            node, [node.key, node.value, self.get_ifs(node)]
        )

    def visit_Lambda(self, node):
        self.visit_defaults_of(node.args)
        return self.visit_scope(node, node.body)

    def visit_scope(self, func_node, content):
        self.current_fn = self.current_fn + [func_node]
        self.visit(content)
        self.current_fn = self.current_fn[:-1]

    @staticmethod
    def get_ifs(node):
        return [comp.ifs for comp in node.generators]

    @staticmethod
    def get_generators(node):
        return [comp.iter for comp in node.generators]


def compute_node_to_containing(tree):
    """
    Compute a dictionary mapping each node in the AST to the list of containing functions.

    Includes function definitions, lambdas, and comprehensions.

    :param tree: The AST to analyze.

    :return: A dictionary mapping each node in the AST to the list of containing functions.
        The last element of the list is the function that directly contains the node.
    """
    to_containing = NodeToContainingFunction()
    to_containing.visit(tree)
    return to_containing.node_to_containing


node_to_is_executed_immediately = {
    ast.Lambda: False,
    ast.FunctionDef: False,
    ast.AsyncFunctionDef: False,
    ast.ListComp: True,
    ast.SetComp: True,
    ast.DictComp: True,
    ast.GeneratorExp: False,
}


def executed_immediately(stack):
    """
    Whether a node that's contained in the given stack is executed immediately.
    """
    return all(node_to_is_executed_immediately[type(x)] for x in stack)


def compute_enclosed_variables(scope_info, pcfg, already_annotated):
    """
    Compute all enclosed variables for the given function.

    :param scope_info: The scope information for the program.
    :param pcfg: The PCFG for the program.
    :param already_annotated: The set of nodes that have already been annotated.

    :return: A tuple of two lists (immediately_executed, closed) where:
        - immediately_executed is the list of variables that are enclosed and executed immediately.
        - closed is the list of variables that are enclosed and placed in a closure.
    """
    node_to_containing = compute_node_to_containing(pcfg.function_astn)

    closed_variables = [
        x
        for x in scope_info
        if x != pcfg.function_astn
        if x not in already_annotated
        and isinstance(scope_info[x], FunctionScope)
        and scope_info[x].function_node == pcfg.function_astn
    ]

    immediately_executed, closed = [], []

    for node in closed_variables:
        if node not in node_to_containing:
            continue
        first, *stack = node_to_containing[node]
        assert first == pcfg.function_astn
        if not stack:
            continue
        if executed_immediately(stack):
            immediately_executed.append(node)
        else:
            closed.append(node)

    return immediately_executed, closed
