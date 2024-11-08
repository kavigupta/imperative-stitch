import re
from typing import List

import neurosym as ns

VARIABLE_REGEX = re.compile(r"var-.*")


class AbstractionHandler(ns.python_def_use_mask.Handler):
    """
    Handler for an abstraction node. This effectively runs through
        the body of the abstraction, pausing at each abstraction variable
        to wait for the next node in the tree to be processed.

    E.g., in an abstraction node (fn_1 (Constant i2 1) &x:0), where fn_1 has body
            (Assign (Name %1 Store) #0)
        the handler would traverse Assign then Name, then pause, awaiting the
            next argument to be passed in. It would then get &x:0 passed in [since
            abstractions are processed in a custom order related to the ordering
            of the arguments' appearances in the body], and continue traversing,
            substituting &x:0 for the passed in node.

    The way this is accomplished is via the traverser field, which performs the
        traversal over the body of the abstraction. See AbstractionBodyTraverser
        for more information.

    handler_fn is used to create the default handler fn in the mask copy used to
        absorb the body of the abstraction. This is necessary because the handler
        could be something like a target handler.

    """

    def __init__(
        self,
        mask,
        defined_production_idxs,
        config,
        head_symbol,
        abstraction,
        position,
        handler_fn=ns.python_def_use_mask.default_handler,
    ):
        super().__init__(mask, defined_production_idxs, config)
        ordering = self.mask.tree_dist.ordering.compute_order(
            self.mask.name_to_id(head_symbol)
        )
        assert ordering is not None, f"No ordering found for {head_symbol}"
        self._traversal_order_stack = ordering[::-1]

        body = ns.to_type_annotated_ns_s_exp(
            abstraction.body, config.dfa, abstraction.dfa_root
        )

        self.traverser = AbstractionBodyTraverser(
            mask,
            config,
            body,
            lambda mask_copy, sym: handler_fn(
                position, sym, mask_copy, self.defined_production_idxs, self.config
            ),
        )

    def __undo__init__(self):
        self.traverser.undo()

    def on_child_enter(
        self, position: int, symbol: int
    ) -> ns.python_def_use_mask.Handler:
        """
        Make sure to collect the children of the abstraction, so it can
            be iterated once the abstraction is fully processed.
        """
        order_pos = self._traversal_order_stack.pop()
        undo_1 = lambda: self._traversal_order_stack.append(order_pos)
        assert order_pos == position, "Incorrect traversal order"
        underlying, undo_2 = self.traverser.last_handler.on_child_enter(
            self.traverser.current_position, symbol
        )
        return CollectingHandler(symbol, underlying), ns.chain_undos([undo_1, undo_2])

    def on_child_exit(
        self, position: int, symbol: int, child: ns.python_def_use_mask.Handler
    ):
        undo_1 = self.traverser.last_handler.on_child_exit(
            self.traverser.current_position, symbol, child
        )
        undo_2 = self.traverser.new_argument(child.node)
        return ns.chain_undos([undo_1, undo_2])

    def is_defining(self, position: int) -> bool:
        return self.traverser.is_defining

    def currently_defined_indices(self) -> list[int]:
        return self.traverser.last_handler.currently_defined_indices()

    @property
    def defined_symbols(self):
        handler = self.traverser.last_handler
        return handler.defined_symbols if hasattr(handler, "defined_symbols") else set()


class AbstractionBodyTraverser:
    """
    This class handles traversal of the body of an abstraction.
        It does so via the the _task_stack, which contains a list of tasks, which are
        either to traverse a node or to exit a node. When a node is traversed, the
        task is popped off the stack, and the node is processed. If the node is a
        variable, we are done with adding an argument. If the node is a symbol, we
        add the symbol to the mask copy, and add the children to the stack.

    We iterate on a copy of the def-use mask, which is important because
        the original mask used to create the AbstractionHandler will be modified
        as the arguments to the abstraction are processed. The copy is created with
        a single handler, which is a default handler for the body.
    """

    def __init__(self, mask, config, body, create_handler):
        self.mask = mask
        self.config = config
        self.create_handler = create_handler

        self._task_stack = [("traverse", body, 0)]
        self._name = None
        self._mask_copy = None
        self._is_defining = None
        self._position = None
        self._variables_to_reuse = {}

        self.undo = self.new_argument(None)

    @property
    def last_handler(self):
        return self._mask_copy.handlers[-1]

    @property
    def current_position(self):
        assert self._position is not None
        return self._position

    @property
    def is_defining(self):
        assert self._is_defining is not None
        return self._is_defining

    def process_until_variable(self):
        undos = []
        while self._task_stack:
            task_type = self._task_stack[-1][0]
            if task_type == "traverse":
                out = self.traverse_body(undos)
                if out is not None:
                    return out, undos
            elif task_type == "exit":
                self.exit(undos)
            else:
                raise ValueError(f"Unrecognized task type {task_type}")
        return None, undos

    def traverse_body(self, undos):
        _, node, position = self._task_stack.pop()
        undos.append(lambda: self._task_stack.append(("traverse", node, position)))
        if VARIABLE_REGEX.match(node.symbol):
            assert (
                self._mask_copy is not None
            ), "We do not support the identity abstraction"
            return self.traverse_variable(node, position, undos)
        sym = self.mask.name_to_id(node.symbol)
        root = self._mask_copy is None
        if root:
            self._mask_copy = self.mask.with_handler(
                lambda mask_copy: self.create_handler(mask_copy, sym)
            )
            undos.append(lambda: setattr(self, "_mask_copy", None))
        else:
            undo = self._mask_copy.on_entry(position, sym)
            undos.append(undo)
        order = self.mask.tree_dist.ordering.order(sym, len(node.children))
        if not root:
            self._task_stack.append(("exit", sym, position))
            undos.append(self._task_stack.pop)
        for i in order[::-1]:
            self._task_stack.append(("traverse", node.children[i], i))
            undos.append(self._task_stack.pop)
        return None

    def traverse_variable(self, node, position, undos):
        # If the node is a variable, check if it is one that has already been processed
        name = node.symbol
        if name in self._variables_to_reuse:
            self._task_stack.append(
                ("traverse", self._variables_to_reuse[name], position)
            )
            undos.append(self._task_stack.pop)
            return None
        is_defining = self._mask_copy.handlers[-1].is_defining(position)
        return is_defining, position, name

    def exit(self, undos):
        _, sym, position = self._task_stack.pop()
        undos.append(lambda: self._task_stack.append(("exit", sym, position)))
        undos.append(self._mask_copy.on_exit(position, sym))

    def new_argument(self, node):
        """
        Iterate through the body of the abstraction, and set the _is_defining and _position values.

        Args:
            node: The node to assign to the last variable. None if we are just starting,
                otherwise the argument that was just processed.
        """
        undos = []
        name = self._name
        if name is not None:
            self._variables_to_reuse[name] = node
            undos.append(lambda: self._variables_to_reuse.pop(name))
        out, undos_rest = self.process_until_variable()
        undos += undos_rest
        if out is None:
            return ns.chain_undos(undos)
        previous = self._is_defining, self._position, self._name

        def undo():
            self._is_defining, self._position, self._name = previous

        undos.append(undo)
        self._is_defining, self._position, self._name = out
        return ns.chain_undos(undos)


class CollectingHandler(ns.python_def_use_mask.Handler):
    """
    Wrapper around another handler that collects the node as it is being created.
    """

    disable_arity_check = False  # for testing purposes only

    def __init__(self, sym, underlying_handler):
        super().__init__(
            underlying_handler.mask,
            underlying_handler.currently_defined_indices(),
            underlying_handler.config,
        )
        self.underlying_handler = underlying_handler
        self.sym: int = sym
        self.children = {}

    @property
    def node(self):
        sym, arity = self.mask.id_to_name_and_arity(self.sym)
        if not self.disable_arity_check:
            assert (
                len(self.children) == arity
            ), f"{sym} expected {arity} children, got {len(self.children)}"
        return ns.SExpression(
            sym, [self.children[i].node for i in range(len(self.children))]
        )

    def on_enter(self):
        self.underlying_handler.on_enter()

    def on_exit(self):
        return self.underlying_handler.on_exit()

    def on_child_enter(
        self, position: int, symbol: int
    ) -> ns.python_def_use_mask.Handler:
        underlying, undo = self.underlying_handler.on_child_enter(position, symbol)
        return CollectingHandler(symbol, underlying), undo

    def on_child_exit(
        self, position: int, symbol: int, child: ns.python_def_use_mask.Handler
    ):
        assert position not in self.children, f"Position {position} already filled"
        self.children[position] = child

        def undo():
            self.children.pop(position)

        undo_2 = self.underlying_handler.on_child_exit(position, symbol, child)
        return ns.chain_undos([undo, undo_2])

    def is_defining(self, position: int) -> bool:
        return self.underlying_handler.is_defining(position)

    @property
    def defined_symbols(self):
        return self.underlying_handler.defined_symbols

    def compute_mask(
        self,
        position: int,
        symbols: List[int],
        idx_to_name: List[str],
        special_case_predicates: List[
            ns.python_def_use_mask.SpecialCaseSymbolPredicate
        ],
    ):
        return self.underlying_handler.compute_mask(
            position, symbols, idx_to_name, special_case_predicates
        )


class AbstractionHandlerPuller(ns.python_def_use_mask.HandlerPuller):
    def __init__(self, abstractions):
        self.abstractions = abstractions

    def pull_handler(
        self, position, symbol, mask, defined_production_idxs, config, handler_fn
    ):
        abstraction = self.abstractions["~".join(symbol.split("~")[:-1])]
        return AbstractionHandler(
            mask,
            defined_production_idxs,
            config,
            symbol,
            abstraction,
            position,
            handler_fn,
        )
