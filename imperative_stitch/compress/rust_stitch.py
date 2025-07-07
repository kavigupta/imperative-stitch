import copy
from dataclasses import dataclass

import neurosym as ns
import stitch_core

from imperative_stitch.compress.abstraction import Abstraction


@dataclass
class CompressionResult:
    abstractions: list[Abstraction]
    rewritten: list[ns.SExpression]


@dataclass
class PartialAbstraction:
    name: str
    body: ns.SExpression
    root_sym: str
    metavar_syms: list[str]
    symvars_syms: list[str]
    choicevar_syms: list[str]

    @property
    def arity(self) -> int:
        return (
            len(self.metavar_syms) + len(self.symvars_syms) + len(self.choicevar_syms)
        )

    def handle_0_arity_leaves(
        self, rewritten: list[ns.SExpression]
    ) -> list[ns.SExpression]:
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

        self.body = handle_0_arity_leaves(self.body)
        rewritten = [handle_0_arity_leaves(rewr) for rewr in rewritten]
        return rewritten

    def extract_symvars(self, rewritten: list[ns.SExpression]) -> list[ns.SExpression]:
        """
        Remove "metavariables" that are symbols (symbol Name) to the symvar_syms.

        In the rewritten code, we need to reorder the call variables to match metavariables followed by symvars.
        """

        assert not self.symvars_syms, "Symvars should be empty at this point."
        assert not self.choicevar_syms, "Symvars should be empty at this point."

        metavariable_indices = []
        symvar_indices = []
        for i, sym in enumerate(self.metavar_syms):
            if sym == "Name":
                symvar_indices.append(i)
            else:
                metavariable_indices.append(i)

        self.metavar_syms, self.symvars_syms = (
            [self.metavar_syms[i] for i in metavariable_indices],
            [self.metavar_syms[i] for i in symvar_indices],
        )

        replace = {
            **{f"#{idx}": f"#{i}" for i, idx in enumerate(metavariable_indices)},
            **{f"#{idx}": f"%{i}" for i, idx in enumerate(symvar_indices, 1)},
        }
        self.body = self.replace_leaves(self.body, replace)

        return [
            self.reorder_call_variables(
                rewr,
                [*metavariable_indices, *symvar_indices],
            )
            for rewr in rewritten
        ]

    def extract_choicevars(
        self, rewritten: list[ns.SExpression]
    ) -> list[ns.SExpression]:
        assert not self.choicevar_syms, "Choicevars should be empty at this point."
        target_vars = sorted(
            {
                x.symbol
                for x in ns.postorder(self.body, leaves=False)
                if is_variable(x.symbol)
            }
        )
        assert all(
            x.startswith("#") for x in target_vars
        ), "only metavariables should appear left of an App"
        indices = [None] * self.arity
        replace = {}
        new_metavars = []
        new_choicevars = []
        current_idx = 0
        num_non_choicevars = (
            len(self.metavar_syms) + len(self.symvars_syms) - len(target_vars)
        )
        for i, sym in enumerate(self.metavar_syms):
            ivar = f"#{i}"
            if ivar in target_vars:
                replace[ivar] = f"?{len(new_choicevars)}"
                indices[num_non_choicevars + len(new_choicevars)] = i
                new_choicevars.append(sym)
            else:
                replace[ivar] = f"#{current_idx}"
                new_metavars.append(sym)
                indices[current_idx] = i
                current_idx += 1
        for i, _ in enumerate(self.symvars_syms):
            indices[current_idx] = i + len(self.metavar_syms)
            current_idx += 1
        self.body = self.replace_leaves(self.body, replace)

        def eta_longify(exp: ns.SExpression) -> ns.SExpression:
            if not isinstance(exp, ns.SExpression):
                assert isinstance(exp, str), "Expected a string or SExpression"
                if exp.startswith("?"):
                    return ns.SExpression("/seq", (exp,))
                return exp
            children = [eta_longify(child) for child in exp.children]
            if is_variable(exp.symbol):
                assert exp.symbol.startswith("?"), "Choicevars should start with '?'"
                return ns.SExpression("/seq", (exp.symbol, *children))
            return ns.SExpression(exp.symbol, children)

        self.body = eta_longify(self.body)

        self.metavar_syms = new_metavars
        self.choicevar_syms = new_choicevars
        return [self.reorder_call_variables(rewr, indices) for rewr in rewritten]

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
            idx = (
                int(var[1:])
                if var.startswith("#")
                else len(self.metavar_syms) + len(self.symvars_syms) + int(var[1:])
            )
            variable_indices_to_pull.append((idx, var.startswith("#")))
        variable_indices_to_pull.sort(key=lambda x: x[0])

        skipped_metavars = []
        rename_map = {}
        count = 0
        for idx in range(len(self.metavar_syms)):
            if f"#{idx}" in prefix_variables:
                skipped_metavars.append(idx)
                continue
            rename_map[f"#{idx}"] = f"#{count}"
            count += 1

        skipped_choicevars = []
        count = 0
        for idx in range(len(self.choicevar_syms)):
            if f"?{idx}" in prefix_variables:
                skipped_choicevars.append(idx)
                continue
            rename_map[f"?{idx}"] = f"?{count}"
            count += 1

        self.metavar_syms = [
            self.metavar_syms[i]
            for i in range(len(self.metavar_syms))
            if i not in skipped_metavars
        ]

        self.choicevar_syms = [
            self.choicevar_syms[i]
            for i in range(len(self.choicevar_syms))
            if i not in skipped_choicevars
        ]

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
                pulled = children.pop(idx)
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

        self.body = self.replace_leaves(
            ns.SExpression("/subseq", self.body.children[len(prefix_variables) :]),
            rename_map,
        )

        rewritten = [rewrite_call_site(rewr) for rewr in rewritten]
        return rewritten

    def to_abstraction(self) -> Abstraction:
        return Abstraction.of(
            name=self.name,
            body=self.body,
            dfa_root=self.root_sym,
            dfa_metavars=self.metavar_syms,
            dfa_symvars=self.symvars_syms,
            dfa_choicevars=self.choicevar_syms,
        )

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
    other_abstractions = result.abstractions
    rewritten = [ns.parse_s_expression(x) for x in result.rewritten]
    while other_abstractions:
        abstr, rewritten, other_abstractions = compute_abstraction(
            other_abstractions[0], rewritten, other_abstractions[1:]
        )
        abstractions.append(abstr)
    return CompressionResult(abstractions, rewritten)


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
        metavar_syms=abstr.tdfa_annotation["metavariable_states"],
        symvars_syms=[],
        choicevar_syms=[],
    )

    s_exprs = rewritten + [ns.parse_s_expression(x.body) for x in other_abstractions]

    s_exprs = partial.handle_0_arity_leaves(s_exprs)
    s_exprs = partial.extract_symvars(s_exprs)
    s_exprs = partial.extract_choicevars(s_exprs)
    s_exprs = partial.handle_variables_at_beginning(s_exprs)

    rewritten = s_exprs[: len(rewritten)]

    other_abstr_new = []

    for other_abstr, rewritten_other in zip(
        other_abstractions, s_exprs[len(rewritten) :]
    ):
        other_abstr = copy.copy(other_abstr)
        other_abstr.body = ns.render_s_expression(rewritten_other)
        other_abstr_new.append(other_abstr)

    return partial.to_abstraction(), rewritten, other_abstr_new


def is_variable(symbol: str) -> bool:
    """
    Check if the symbol is a variable.
    """
    return symbol.startswith("%") or symbol.startswith("#") or symbol.startswith("?")
