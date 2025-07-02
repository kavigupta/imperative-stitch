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
        return ns.SExpression(s_exp.symbol, children)


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

    s_exprs = partial.extract_symvars(s_exprs)

    rewritten = s_exprs[: len(rewritten)]

    other_abstr_new = []

    for other_abstr, rewritten_other in zip(
        other_abstractions, s_exprs[len(rewritten) :]
    ):
        other_abstr = copy.copy(other_abstr)
        other_abstr.body = ns.render_s_expression(rewritten_other)
        other_abstr_new.append(other_abstr)

    return partial.to_abstraction(), rewritten, other_abstr_new
