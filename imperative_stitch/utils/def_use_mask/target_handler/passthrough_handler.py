import ast

from ..handler import Handler


class PassthroughLHSHandler(Handler):
    """
    Pass through handler that does not collect any information,
        instead it just targets the children at the given indices.

    If indices is None, it will target all children.
    """

    def __init__(self, mask, valid_symbols, config, indices=None):
        super().__init__(mask, valid_symbols, config)
        self.defined_symbols = set()
        self.indices = indices

    def child_is_targeted(self, position: int) -> bool:
        return self.indices is None or position in self.indices

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if self.child_is_targeted(position):
            return self.target_child(symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        if self.child_is_targeted(position):
            self.defined_symbols |= child.defined_symbols

    def is_defining(self, position: int) -> bool:
        return True


assert ast.Starred._fields == ("value", "ctx")
StarredHandler = lambda mask, valid_symbols, config: PassthroughLHSHandler(
    mask, valid_symbols, config, indices=[0]
)