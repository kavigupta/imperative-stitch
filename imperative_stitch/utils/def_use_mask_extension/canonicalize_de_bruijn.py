import copy
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import neurosym as ns

from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies,
)
from imperative_stitch.utils.classify_nodes import SYMBOL_TYPES
from imperative_stitch.utils.def_use_mask_extension.mask import def_use_mask
from imperative_stitch.utils.def_use_mask_extension.ordering import (
    PythonWithAbstractionsNodeOrdering,
)
from imperative_stitch.utils.dsl_with_abstraction import add_abstractions
from imperative_stitch.utils.types import SEPARATOR, get_dfa_state

dbv_type = "DBV"


def dbvar_symbol(idx):
    return f"dbvar-{idx}{SEPARATOR}{dbv_type}"


dbvar_symbol_regex = re.compile(
    r"dbvar-(?P<which>\d+|successor)"
    + "("
    + re.escape(SEPARATOR)
    + re.escape(dbv_type)
    + ")?"
)

dbvar_successor_symbol = dbvar_symbol("successor")

dbvar_wrapper_symbol_by_root_type = {
    root_type: f"dbvar{SEPARATOR}{root_type}" for root_type in SYMBOL_TYPES
}


def is_dbvar_wrapper_symbol(symbol):
    return symbol in dbvar_wrapper_symbol_by_root_type.values()


def create_de_bruijn_child(idx, max_explicit_dbvar_index):
    """
    Create a de bruijn child. to be placed into a wrapper.

    Series, starting at (dbvar-0~DBV) and continuing to (dbvar-N~DBV),
        where N is the max_explicit_dbvar_index, then continuing
        with (dbvar-successor~DBV (dbvar-N~DBV)),
        (dbvar-successor (dbvar-successor~DBV (dbvar-N~DBV))), etc.
    """
    if idx <= max_explicit_dbvar_index:
        return ns.SExpression(dbvar_symbol(idx), ())
    return ns.SExpression(
        dbvar_successor_symbol,
        (create_de_bruijn_child(idx - 1, max_explicit_dbvar_index),),
    )


def create_de_bruijn(idx, max_explicit_dbvar_index, dfa_sym):
    """
    Create a de bruijn name. This is a wrapper around a de bruijn child, and looks e.g., like
        (dbvar~Name (dbvar-0~DBV)).

    The root type is given by dfa_sym.
    """
    return ns.SExpression(
        dbvar_wrapper_symbol_by_root_type[dfa_sym],
        (create_de_bruijn_child(idx, max_explicit_dbvar_index),),
    )


def get_idx(s_exp_de_bruijn):
    """
    Get the index of the given de bruijn variable.
    """
    if is_dbvar_wrapper_symbol(s_exp_de_bruijn.symbol):
        assert len(s_exp_de_bruijn.children) == 1, s_exp_de_bruijn
        return get_idx(s_exp_de_bruijn.children[0])
    mat = dbvar_symbol_regex.match(s_exp_de_bruijn.symbol)
    assert mat, s_exp_de_bruijn.symbol
    after = mat.group(1)
    if after == "successor":
        assert len(s_exp_de_bruijn.children) == 1
        return get_idx(s_exp_de_bruijn.children[0]) + 1
    assert len(s_exp_de_bruijn.children) == 0
    return int(after)


def canonicalize_de_bruijn_batched(
    programs,
    root_states,
    dfa,
    abstrs,
    max_explicit_dbvar_index,
    include_abstr_exprs=False,
):
    """
    Convert the programs to a de bruijn representation. Creates a tree distribution
        and then calls the canonicalize_de_bruijn_from_tree_dist function.
    """

    check_have_all_abstrs(dfa, abstrs)

    subset = ns.PythonDSLSubset()
    s_exps = subset.add_programs(dfa, *programs, root=root_states)
    abstr_s_exps = add_abstractions(subset, dfa, *abstrs)
    if include_abstr_exprs:
        s_exps += abstr_s_exps
        root_states = list(root_states) + [abstr.dfa_root for abstr in abstrs]

    dsl_by_root = {
        root: ns.create_python_dsl(dfa, subset, root) for root in set(root_states)
    }
    fam = {
        root: ns.BigramProgramDistributionFamily(
            dsl,
            additional_preorder_masks=[
                lambda dist, dsl: def_use_mask(dist, dsl, dfa=dfa, abstrs=abstrs)
            ],
            include_type_preorder_mask=False,
            node_ordering=lambda dist: PythonWithAbstractionsNodeOrdering(dist, abstrs),
        ).tree_distribution_skeleton
        for root, dsl in dsl_by_root.items()
    }
    return [
        canonicalize_de_bruijn_from_tree_dist(
            fam[root], s_exp, max_explicit_dbvar_index
        )
        for s_exp, root in zip(s_exps, root_states)
    ]


def canonicalize_de_bruijn(program, root_state, dfa, abstrs, max_explicit_dbvar_index):
    """
    Like canonicalize_de_bruijn_batched, but for a single program.
    """
    [result] = canonicalize_de_bruijn_batched(
        [program], [root_state], dfa, abstrs, max_explicit_dbvar_index
    )
    return result


def check_have_all_abstrs(dfa, abstrs):
    """
    Check that all abstractions are present in the given DFA.
    """
    abstr_names = {abstr.name for abstr in abstrs}
    for vs in dfa.values():
        for v in vs:
            if not v.startswith("fn_"):
                continue
            assert (
                v in abstr_names
            ), f"Missing abstraction {v} in abstrs. Have {sorted(abstr_names)}"


def canonicalize_de_bruijn_from_tree_dist(tree_dist, s_exp, max_explicit_dbvar_index):
    """
    Convert the program to a de bruijn representation, given the tree distribution.
    """
    id_to_new = {}
    for node, _, mask in ns.collect_preorder_symbols(s_exp, tree_dist):
        node_sym = tree_dist.symbol_to_index[node.symbol]
        mat = ns.python_def_use_mask.NAME_REGEX.match(node.symbol)
        if not mat:
            continue
        assert not node.children
        currently_defined = get_defined_indices(mask)
        if node_sym not in currently_defined:
            de_bruijn_idx = 0
        else:
            de_bruijn_idx = len(currently_defined) - currently_defined.index(node_sym)
        id_to_new[id(node)] = create_de_bruijn(
            de_bruijn_idx, max_explicit_dbvar_index, mat.group("dfa_sym")
        )

    return s_exp.replace_nodes_by_id(id_to_new)


def get_defined_indices(mask):
    """
    Get the set of defined indices for a given mask.
    """
    mask = get_def_use_chain_mask(mask)
    currently_defined = mask.currently_defined_indices()
    return currently_defined


def get_def_use_chain_mask(mask):
    assert isinstance(mask, ns.python_def_use_mask.DefUseChainPreorderMask)
    return mask


def uncanonicalize_de_bruijn(dfa, s_exp_de_bruijn, abstrs):
    """
    Uncanonicalize the de bruijn representation, replacing variables
        with names __0, __1, etc.
    """
    if isinstance(s_exp_de_bruijn, str):
        s_exp_de_bruijn = ns.parse_s_expression(s_exp_de_bruijn)
    else:
        assert isinstance(s_exp_de_bruijn, ns.SExpression)
        s_exp_de_bruijn = copy.deepcopy(s_exp_de_bruijn)

    abstrs_dict = {abstr.name: abstr for abstr in abstrs}
    abstr_bodies = [
        ns.to_type_annotated_ns_s_exp(
            abstraction_calls_to_bodies(abstr.body, abstrs_dict), dfa, abstr.dfa_root
        )
        for abstr in abstrs
    ]

    dsl = ns.create_python_dsl(
        dfa,
        ns.PythonDSLSubset.from_s_exps([s_exp_de_bruijn] + abstr_bodies),
        get_dfa_state(s_exp_de_bruijn.symbol),
        add_additional_productions=add_dbvar_additional_productions,
    )
    fam = ns.BigramProgramDistributionFamily(
        dsl,
        additional_preorder_masks=[
            lambda dist, dsl: def_use_mask(dist, dsl, dfa=dfa, abstrs=abstrs)
        ],
        include_type_preorder_mask=False,
        node_ordering=lambda dist: PythonWithAbstractionsNodeOrdering(dist, abstrs),
    )
    tree_dist = fam.tree_distribution_skeleton

    count_vars = 0

    def replace_de_bruijn(node, mask, typ):
        """
        Compute the replacement for a de bruijn node.
        """
        nonlocal count_vars
        indices = get_defined_indices(mask)
        idx = get_idx(node)
        if idx == 0:
            result = ns.SExpression(
                ns.python_def_use_mask.canonicalized_python_name_as_leaf(
                    count_vars, typ
                ),
                (),
            )
            count_vars += 1
            return result
        sym_idx = indices[-idx]
        sym = get_def_use_chain_mask(mask).id_to_name(sym_idx)
        return ns.SExpression(sym, ())

    id_to_new = {}

    def traverse_replacer(node, mask, _1, _2):
        if is_dbvar_wrapper_symbol(node.symbol):
            assert len(node.children) == 1
            new_node = replace_de_bruijn(
                node.children[0], mask, node.symbol.split(SEPARATOR)[-1]
            )
            id_to_new[id(node)] = new_node
            return new_node
        return node

    list(
        ns.collect_preorder_symbols(
            s_exp_de_bruijn,
            tree_dist,
            replace_node_midstream=traverse_replacer,
            symbol_to_index_fn=lambda mask, x: get_def_use_chain_mask(mask).name_to_id(
                x
            ),
        )
    )

    return s_exp_de_bruijn.replace_nodes_by_id(id_to_new)


def compute_de_bruijn_limit(tree_dist: ns.TreeDistribution) -> int:
    """
    Compute the de bruijn limit for the given tree distribution.
    """
    dbvars = []
    for x, _ in tree_dist.symbols:
        mat = dbvar_symbol_regex.match(x)
        if mat and mat.group(1) != "successor":
            dbvars.append(int(mat.group("which")))
    if not dbvars:
        return 0
    return max(dbvars)


@dataclass
class DeBruijnMaskState:
    """
    Handles the mask behavior of de bruijin nodes.
    """

    max_explicit_dbvar_index: int
    is_defn: bool
    num_available_symbols: int

    def one_less_symbol(self) -> "DeBruijnMaskState":
        return DeBruijnMaskState(
            self.max_explicit_dbvar_index, self.is_defn, self.num_available_symbols - 1
        )


class DeBruijnHandler(ns.python_def_use_mask.Handler):
    def __init__(self, mask, defined_production_idxs, config, state, dbvar_components):
        super().__init__(mask, defined_production_idxs, config)
        self.state = state
        self.dbvar_components = dbvar_components

    def is_defining(self, position: int) -> bool:
        raise NotImplementedError

    def compute_mask(
        self,
        position: int,
        symbols: List[int],
        idx_to_name: List[str],
        special_case_predicates: List[
            ns.python_def_use_mask.SpecialCaseSymbolPredicate
        ],
    ):
        """
        Compute the mask for the given symbols.
        """
        del position, idx_to_name, special_case_predicates
        mask = self.compute_dict_mask()
        mask = [mask.get(self.mask.tree_dist.symbols[sym][0], False) for sym in symbols]
        return mask

    @abstractmethod
    def compute_dict_mask(self):
        pass

    def on_child_enter(
        self, position: int, symbol: int
    ) -> ns.python_def_use_mask.Handler:
        state = self.state
        if self.mask.tree_dist.symbols[symbol][0] == dbvar_successor_symbol:
            state = state.one_less_symbol()
            self.dbvar_components.append(1)
            typ = DeBruijnVarSuccessorHandler
        else:
            index = int(
                self.mask.tree_dist.symbols[symbol][0].split("-")[-1].split("~")[0]
            )
            self.dbvar_components.append(index)
            typ = DeBruijnVarHandler

        handler = typ(
            self.mask,
            self.defined_production_idxs,
            self.config,
            state,
            self.dbvar_components,
        )
        return handler, self.dbvar_components.pop


class DeBruijnVarHandler(DeBruijnHandler):
    def compute_dict_mask(self):
        raise NotImplementedError

    def on_child_exit(
        self, position: int, symbol: int, child: ns.python_def_use_mask.Handler
    ):
        return lambda: None


class DeBruijnVarSuccessorHandler(DeBruijnHandler):
    def compute_dict_mask(self):
        mask = {}
        # We are inside a successor, so the de bruijn limit is valid, as is successor
        mask[dbvar_symbol(self.state.max_explicit_dbvar_index)] = True
        if self.state.num_available_symbols > self.state.max_explicit_dbvar_index:
            mask[dbvar_successor_symbol] = True
        return mask

    def on_child_exit(
        self, position: int, symbol: int, child: ns.python_def_use_mask.Handler
    ):
        return lambda: None


class DBVarWrapperHandler(DeBruijnHandler, ns.python_def_use_mask.TargetHandler):
    def __init__(
        self,
        mask,
        defined_production_idxs,
        config,
        max_explicit_dbvar_index: int,
        num_available_symbols: int,
        is_defn: bool,
    ):
        ns.python_def_use_mask.TargetHandler.__init__(
            self, mask, defined_production_idxs, config
        )
        DeBruijnHandler.__init__(
            self,
            mask,
            defined_production_idxs,
            config,
            DeBruijnMaskState(max_explicit_dbvar_index, is_defn, num_available_symbols),
            [],
        )
        self.num_available_symbols = num_available_symbols

    def compute_dict_mask(self):
        mask = {}
        # We are not inside a successor, so we need to iterate upwards
        start_at = 0 if self.state.is_defn else 1
        for i in range(start_at, self.state.num_available_symbols + 1):
            if i > self.state.max_explicit_dbvar_index:
                mask[dbvar_successor_symbol] = True
                break
            mask[dbvar_symbol(i)] = True
        return mask

    def on_child_exit(
        self, position: int, symbol: int, child: ns.python_def_use_mask.Handler
    ):
        symbol = ns.python_def_use_mask.ExtraVar(
            self.num_available_symbols - sum(self.dbvar_components)
        )
        if symbol is not None and symbol not in self.defined_symbols:
            self.defined_symbols.append(symbol)
            return self.defined_symbols.pop
        return lambda: None


def dsl_subset_from_dbprograms(*programs, roots, dfa, abstrs, max_explicit_dbvar_index):
    assert len(programs) == len(roots), "The number of programs and roots must match."

    programs_all = canonicalize_de_bruijn_batched(
        programs,
        roots,
        dfa,
        abstrs,
        max_explicit_dbvar_index,
        include_abstr_exprs=True,
    )
    subset = ns.PythonDSLSubset.from_s_exps(programs_all)
    return programs_all[: len(programs)], subset


def add_dbvar_additional_productions(dslf):
    dslf.concrete(dbvar_successor_symbol, "DBV -> DBV", None)
    for root_type, sym in dbvar_wrapper_symbol_by_root_type.items():
        dslf.concrete(sym, f"DBV -> {root_type}", None)


class DBVarSymbolPredicate(ns.python_def_use_mask.SpecialCaseSymbolPredicate):
    """
    Predicate that applies to the DBVar symbol.
    """

    def __init__(self, tree_dist: ns.TreeDistribution):
        super().__init__(tree_dist)
        self.dbvars = [
            is_dbvar_wrapper_symbol(symbol) for symbol, _ in tree_dist.symbols
        ]

    def applies(self, symbol: int) -> bool:
        return self.dbvars[symbol]

    def compute(self, symbol: int, names: List[str]) -> bool:
        return len(names) > 0


class DBVarHandlerPuller(ns.python_def_use_mask.HandlerPuller):
    def __init__(self):
        pass

    def pull_handler(
        self, position, symbol, mask, defined_production_idxs, config, handler_fn
    ):
        return DBVarWrapperHandler(
            mask,
            defined_production_idxs,
            config,
            compute_de_bruijn_limit(mask.tree_dist),
            len(mask.currently_defined_indices()),
            mask.handlers[-1].is_defining(position),
        )
