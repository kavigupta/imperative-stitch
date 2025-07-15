import copy
import json
from dataclasses import dataclass
from typing import List

import neurosym as ns
import stitch_core

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.utils.classify_nodes import SYMBOL_TYPES


@dataclass
class CompressionResult:
    abstractions: list[Abstraction]
    rewritten: list[ns.SExpression]


@dataclass
class PartialAbstraction:
    name: str
    body: ns.SExpression
    root_sym: str
    symbols_each: list[str]
    kinds_each: list[str]

    @property
    def arity(self) -> int:
        return len(self.symbols_each)

    def handle_rooted_non_eta_long(
        self, rewritten: list[ns.SExpression]
    ) -> list[ns.SExpression]:
        def handle_rooted_non_eta_long(
            exp: ns.SExpression,
        ) -> ns.SExpression:
            if not isinstance(exp, ns.SExpression):
                return exp
            children = [handle_rooted_non_eta_long(child) for child in exp.children]
            if exp.symbol == self.name:
                if len(children) == self.arity:
                    return ns.SExpression(exp.symbol, children)
                assert self.root_sym == "seqS", "only valid non-eta-long root is seqS"
                return ns.SExpression(
                    "/seq",
                    [
                        ns.SExpression(
                            "/splice",
                            [ns.SExpression(exp.symbol, children[: self.arity])],
                        ),
                        *children[self.arity :],
                    ],
                )
            return ns.SExpression(exp.symbol, children)

        return [handle_rooted_non_eta_long(rewr) for rewr in rewritten]

    def extract_symvars(self, rewritten: list[ns.SExpression]) -> list[ns.SExpression]:
        """
        Remove "metavariables" that are symbols (symbol Name) to the symvar_syms.

        In the rewritten code, we need to reorder the call variables to match metavariables followed by symvars.
        """

        for i, sym in enumerate(self.symbols_each):
            if sym not in SYMBOL_TYPES:
                continue
            assert self.kinds_each[i] == "#", "Should be a metavariable at this point."
            self.kinds_each[i] = "%"
        return rewritten

    def extract_choicevars(
        self, rewritten: list[ns.SExpression]
    ) -> list[ns.SExpression]:
        target_vars = sorted(
            {
                x.symbol
                for x in ns.postorder(self.body, leaves=False)
                if is_variable(x.symbol)
            }
            | {f"#{i}" for i, x in enumerate(self.symbols_each) if x == "seqS"}
        )
        assert all(
            x.startswith("#") for x in target_vars
        ), "only metavariables should appear left of an App"

        for var in target_vars:
            self.kinds_each[int(var[1:])] = "?"

        def eta_longify(exp: ns.SExpression) -> ns.SExpression:
            if not isinstance(exp, ns.SExpression):
                assert isinstance(exp, str), "Expected a string or SExpression"
                if exp in target_vars:
                    return ns.SExpression("/seq", (exp,))
                return exp
            children = [eta_longify(child) for child in exp.children]
            if is_variable(exp.symbol):
                assert exp.symbol in target_vars
                return ns.SExpression("/seq", (exp.symbol, *children))
            return ns.SExpression(exp.symbol, children)

        self.body = eta_longify(self.body)

        return rewritten

    def handle_variables_at_beginning(
        self, rewritten: list[ns.SExpression]
    ) -> list[ns.SExpression]:
        """
        Take any variable at the beginning of the whole sequence, and remove it, putting it in
        each call site. Use `/subseq` in the body to indicate this, and /splice at the call sites.
        """
        if self.body.symbol != "/seq":
            return rewritten

        prefix_variables = []
        for child in self.body.children:
            if not isinstance(child, str) or not is_variable(child):
                break
            if child in prefix_variables:
                prefix_variables = prefix_variables[: prefix_variables.index(child)]
                break
            prefix_variables.append(child)
        reused_vars = [
            prefix_variables.index(node)
            for x in self.body.children[len(prefix_variables) :]
            for node in ns.postorder(x)
            if node in prefix_variables
        ]

        if reused_vars:
            prefix_variables = prefix_variables[: min(reused_vars)]

        if not prefix_variables:
            return rewritten

        variable_indices_to_pull = []
        for var in prefix_variables:
            idx = int(var[1:])
            variable_indices_to_pull.append((idx, self.kinds_each[idx] == "#"))
        variable_indices_to_pull.sort(key=lambda x: x[0])

        for idx, _ in variable_indices_to_pull:
            self.kinds_each[idx] = None

        def rewrite_call_site(
            exp: ns.SExpression,
        ) -> ns.SExpression:
            if not isinstance(exp, ns.SExpression):
                return exp
            children = [rewrite_call_site(child) for child in exp.children]
            if exp.symbol != self.name:
                return ns.SExpression(exp.symbol, children)
            prefix = []
            # reverse order so pop works
            for idx, is_metavar in variable_indices_to_pull[::-1]:
                pulled = children[idx]
                if not is_metavar:
                    pulled = ns.SExpression("/splice", (pulled,))
                prefix = [pulled] + prefix
            return ns.SExpression(
                "/seq",
                [
                    *prefix,
                    ns.SExpression("/splice", [ns.SExpression(self.name, children)]),
                ],
            )

        self.body = ns.SExpression(
            "/subseq", self.body.children[len(prefix_variables) :]
        )

        rewritten = [rewrite_call_site(rewr) for rewr in rewritten]
        return rewritten

    def to_abstraction(
        self, rewritten: List[ns.SExpression]
    ) -> tuple[Abstraction, list[ns.SExpression]]:
        metavar_indices = []
        symvar_indices = []
        choicevar_indices = []
        for i, kind in enumerate(self.kinds_each):
            if kind == "#":
                metavar_indices.append(i)
            elif kind == "?":
                choicevar_indices.append(i)
            elif kind == "%":
                symvar_indices.append(i)
            else:
                assert kind == None

        all_indices = metavar_indices + symvar_indices + choicevar_indices
        remapping_dict = {
            f"#{index}": f"{self.kinds_each[index]}{idx_within + (1 if self.kinds_each[index] == '%' else 0)}"
            for indices in (metavar_indices, symvar_indices, choicevar_indices)
            for idx_within, index in enumerate(indices)
        }

        body = self.replace_leaves(self.body, remapping_dict)
        rewritten = [
            self.reorder_call_variables(rewr, all_indices) for rewr in rewritten
        ]

        abstr = Abstraction.of(
            name=self.name,
            body=body,
            dfa_root=self.root_sym,
            dfa_metavars=[self.symbols_each[i] for i in metavar_indices],
            dfa_symvars=[self.symbols_each[i] for i in symvar_indices],
            dfa_choicevars=[self.symbols_each[i] for i in choicevar_indices],
        )
        return abstr, rewritten

    def reorder_call_variables(
        self, rewritten: ns.SExpression, indices: list[int]
    ) -> ns.SExpression:
        """
        Reorder the call variables in the rewritten code to match the given indices.
        """
        if not isinstance(rewritten, ns.SExpression):
            return rewritten
        children = [
            self.reorder_call_variables(arg, indices) for arg in rewritten.children
        ]
        if rewritten.symbol != self.name:
            return ns.SExpression(rewritten.symbol, children)
        assert len(children) >= len(indices), "Not enough children to reorder."
        indices = indices + list(range(len(children), len(rewritten.children)))
        new_children = [children[i] for i in indices]
        return ns.SExpression(rewritten.symbol, new_children)

    def replace_leaves(
        self, s_exp: ns.SExpression, replacements: dict[str, str]
    ) -> ns.SExpression:
        """
        Replace leaves in the SExpression with the given replacements.
        """
        if not isinstance(s_exp, ns.SExpression):
            return replacements.get(s_exp, s_exp)
        children = [
            self.replace_leaves(child, replacements) for child in s_exp.children
        ]
        return ns.SExpression(replacements.get(s_exp.symbol, s_exp.symbol), children)


def process_rust_stitch(
    result: stitch_core.CompressionResult,
) -> CompressionResult:
    # TODO implement
    abstractions = []
    other_abstractions = copy.deepcopy(result.abstractions)
    for abstr in other_abstractions:
        abstr.body = ns.render_s_expression(
            handle_0_arity_leaves(ns.parse_s_expression(abstr.body))
        )
    rewritten = [
        handle_0_arity_leaves(ns.parse_s_expression(x)) for x in result.rewritten
    ]
    while other_abstractions:
        abstr, rewritten, other_abstractions = compute_abstraction(
            other_abstractions[0], rewritten, other_abstractions[1:]
        )
        abstractions.append(abstr)
    return CompressionResult(abstractions, rewritten)


def handle_0_arity_leaves(
    exp: ns.SExpression,
) -> ns.SExpression:
    if isinstance(exp, ns.SExpression):
        return ns.SExpression(
            exp.symbol,
            [handle_0_arity_leaves(child) for child in exp.children],
        )
    assert isinstance(exp, str), "Expected a string or SExpression"
    if exp in {"/seq"}:
        return ns.SExpression(exp, [])
    return exp


def handle_splice_seqs_in_list_context(
    head: str, children: list[ns.SExpression]
) -> tuple[str, list[ns.SExpression]]:
    new_children = []
    for child in children:
        if isinstance(child, ns.SExpression) and child.symbol == "/splice":
            assert len(child.children) == 1
            splice_child = child.children[0]
            if not isinstance(splice_child, ns.SExpression):
                splice_child = ns.SExpression(splice_child, ())
            if splice_child.symbol == "/seq":
                new_children.extend(splice_child.children)
                continue
            if splice_child.symbol.startswith("fn_"):
                # If the splice is a function call, we need to keep it as a splice.
                new_children.append(child)
                continue
            if splice_child.symbol.startswith("#"):
                # If the splice is a metavariable, we need to keep it as a splice.
                assert not new_children, "Should not have any children at this point."
                head = splice_child.symbol
                new_children.extend(splice_child.children)
                continue
            raise ValueError(
                f"Unexpected splice child: {ns.render_s_expression(splice_child)}"
            )
        else:
            new_children.append(child)
    return head, new_children


def handle_splice_seqs(s_expr: ns.SExpression) -> ns.SExpression:
    if isinstance(s_expr, str):
        return s_expr
    symbol, children = s_expr.symbol, [
        handle_splice_seqs(child) for child in s_expr.children
    ]
    if symbol == "/seq":
        return ns.SExpression(*handle_splice_seqs_in_list_context(symbol, children))
    return ns.SExpression(symbol, children)


def compute_abstraction(
    abstr: stitch_core.Abstraction,
    rewritten: list[ns.SExpression],
    other_abstractions: list[stitch_core.Abstraction],
) -> tuple[Abstraction, list[ns.SExpression], list[stitch_core.Abstraction]]:
    """
    Compute a single abstraction from the given abstraction and the rewritten code.
    """

    partial = PartialAbstraction(
        name=abstr.name,
        body=ns.parse_s_expression(abstr.body),
        root_sym=abstr.tdfa_annotation["root_state"],
        symbols_each=abstr.tdfa_annotation["metavariable_states"],
        kinds_each=["#"] * len(abstr.tdfa_annotation["metavariable_states"]),
    )

    s_exprs = rewritten + [ns.parse_s_expression(x.body) for x in other_abstractions]

    s_exprs = partial.handle_rooted_non_eta_long(s_exprs)
    s_exprs = partial.extract_symvars(s_exprs)
    s_exprs = partial.extract_choicevars(s_exprs)
    s_exprs = partial.handle_variables_at_beginning(s_exprs)

    this_abstr, s_exprs = partial.to_abstraction(s_exprs)
    s_exprs = [handle_splice_seqs(s_expr) for s_expr in s_exprs]

    rewritten = s_exprs[: len(rewritten)]

    other_abstr_new = []

    for other_abstr, rewritten_other in zip(
        other_abstractions, s_exprs[len(rewritten) :]
    ):
        other_abstr = copy.copy(other_abstr)
        other_abstr.body = ns.render_s_expression(rewritten_other)
        other_abstr_new.append(other_abstr)

    return this_abstr, rewritten, other_abstr_new


def is_variable(symbol: str) -> bool:
    """
    Check if the symbol is a variable.
    """
    return symbol.startswith("%") or symbol.startswith("#") or symbol.startswith("?")


def compress_stitch(pythons, *, use_symvars=True, **kwargs) -> CompressionResult:
    cost_prim = {
        "Module": 0,
        "Name": 0,
        "Load": 0,
        "Store": 0,
        "None": 0,
        "list": 0,
        "nil": 0,
        "semi": 0,
        "Constant": 0,
        "Attribute": 0,
        "_slice_content": 0,
        "_slice_slice": 0,
        "_slice_tuple": 0,
        "_starred_content": 0,
        "_starred_starred": 0,
        "/choiceseq": 0,
        "Subscript": 0,
        "Expr": 0,
        "Call": 0,
        "Assign": 0,
        "AugAssign": 0,
        "BinOp": 0,
        "UnaryOp": 0,
        "/seq": 0,
        "FunctionDef": 0,
        "arguments": 0,
        "arg": 0,
        "Compare": 0,
        "Import": 0,
        "Alias": 0,
        "Return": 0,
    }
    s_exps, symbols = convert_all_to_annotated_s_exps(pythons)
    for symbol in symbols:
        symbol_trimmed = symbol.split(ns.python_dsl.names.PYTHON_DSL_SEPARATOR)[0]
        if symbol_trimmed in cost_prim:
            cost_prim[symbol] = cost_prim[symbol_trimmed]
    kwargs = kwargs.copy()
    if use_symvars:
        kwargs["symvar_prefix"] = "&"
    compressed = stitch_core.compress(
        s_exps,
        cost_prim=json.dumps(cost_prim).replace(" ", ""),
        tdfa_json_path="../neurosym-lib/test_data/dfa.json",
        tdfa_root="M",
        valid_metavars='["S","E","seqS"]',
        valid_roots='["S","E","seqS"]',
        tdfa_non_eta_long_states='{"seqS":"S"}',
        tdfa_split=ns.python_dsl.names.PYTHON_DSL_SEPARATOR,
        **kwargs,
    )
    return process_rust_stitch(compressed)


def convert_all_to_annotated_s_exps(pythons):
    s_exps = [
        ns.python_to_type_annotated_ns_s_exp(
            code_snippet,
            ns.python_dfa(),
            "M",
            no_leaves=False,
            only_for_nodes={"None", "Tuple"},
        )
        for code_snippet in pythons
    ]
    symbols = {
        node if isinstance(node, str) else node.symbol
        for program in s_exps
        for node in ns.postorder(program, leaves=True)
    }
    s_exps = [ns.render_s_expression(exp) for exp in s_exps]
    return s_exps, sorted(symbols)
