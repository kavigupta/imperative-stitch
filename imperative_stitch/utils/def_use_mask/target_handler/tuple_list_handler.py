from ..handler import Handler


class TupleListLHSHandler(Handler):
    """
    This is for LHS values where nothing is actually being defined (e.g., Subscript, Attribute, etc.)
    """

    fields = {"elts": 0}

    def __init__(self, mask, valid_symbols, config):
        super().__init__(mask, valid_symbols, config)
        self.defined_symbols = set()

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if position == self.fields["elts"]:
            return self.target_child(symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        if position == self.fields["elts"]:
            self.defined_symbols |= child.defined_symbols

    def is_defining(self, position: int) -> bool:
        return position == self.fields["elts"]