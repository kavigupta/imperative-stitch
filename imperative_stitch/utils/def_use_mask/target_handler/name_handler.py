from ..handler import Handler


class NameTargetHandler(Handler):
    # this works for Name, arg, and Starred
    fields = {"id": 0, "arg": 0, "value": 0}

    def __init__(self, mask, valid_symbols, config):
        super().__init__(mask, valid_symbols, config)
        self.defined_symbols = set()

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if position == self.fields["id"]:
            self.defined_symbols.add(symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        pass

    def is_defining(self, position: int) -> bool:
        return position == self.fields["id"]