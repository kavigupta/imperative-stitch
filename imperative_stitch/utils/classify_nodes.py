import copy
import json

import neurosym as ns
from frozendict import frozendict

SYMBOL_TYPES = ("Name", "NullableName", "NameStr", "NullableNameStr")


def export_dfa(*, abstrs=frozendict({})):
    """
    Takes a transition dictionary of the form above and converts
        it to a dict[state, dict[tag, list[state]]].

    """

    if isinstance(abstrs, (list, tuple)):
        abstrs = {x.name: x for x in abstrs}

    assert isinstance(abstrs, (dict, frozendict)), f"expected dict, got {abstrs}"

    result = copy.deepcopy(ns.python_dfa())

    for k, abstr in abstrs.items():
        assert k == abstr.name, (k, abstr.name)
        result[abstr.dfa_root][k] = (
            abstr.dfa_metavars + abstr.dfa_symvars + abstr.dfa_choicevars
        )

    return result


if __name__ == "__main__":
    print(json.dumps(export_dfa(), indent=2))
