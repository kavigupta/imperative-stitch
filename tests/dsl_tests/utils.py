import neurosym as ns

from imperative_stitch.parser import converter
from imperative_stitch.utils.classify_nodes import export_dfa
from imperative_stitch.utils.def_use_mask import DefUseChainPreorderMask
from imperative_stitch.utils.def_use_mask.ordering import PythonNodeOrdering
from imperative_stitch.utils.export_as_dsl import (
    DSLSubset,
    create_dsl,
    create_smoothing_mask,
)


def fit_to(
    programs,
    parser=ns.python_to_python_ast,
    root="M",
    abstrs=(),
    use_def_use=True,
    use_node_ordering=True,
    smoothing=True,
    include_type_preorder_mask=True,
):
    """
    Set include_type_preorder_mask to False to disable the type preorder mask,
        this is basically only useful in the specific context where we are testing
        the names mask and want no other masks to be applied.
    """
    dfa = export_dfa(abstrs=abstrs)
    programs = [parser(p) for p in programs]
    dsl = create_dsl(
        dfa, DSLSubset.from_program(dfa, *programs, root=root, abstrs=abstrs), root
    )
    dsl_subset = create_dsl(
        dfa, DSLSubset.from_program(dfa, *programs, root=root), root
    )
    smooth_mask = create_smoothing_mask(dsl, dsl_subset)
    apms = [
        lambda dist, dsl: DefUseChainPreorderMask(dist, dsl, dfa=dfa, abstrs=abstrs)
    ]
    node_ordering = (
        (lambda dist: PythonNodeOrdering(dist, abstrs))
        if use_node_ordering
        else ns.DefaultNodeOrdering
    )
    fam = ns.BigramProgramDistributionFamily(
        dsl,
        additional_preorder_masks=apms if use_def_use else [],
        include_type_preorder_mask=include_type_preorder_mask,
        node_ordering=node_ordering,
    )
    counts = fam.count_programs(
        [
            [
                converter.to_type_annotated_ns_s_exp(program, dfa, root)
                for program in programs
            ]
        ]
    )
    dist = fam.counts_to_distribution(counts)[0]
    if smoothing:
        dist = dist.bound_minimum_likelihood(1e-4, smooth_mask)
    return dfa, dsl, fam, dist