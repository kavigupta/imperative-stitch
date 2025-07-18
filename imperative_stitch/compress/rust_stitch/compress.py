import json
import tempfile
from functools import lru_cache

import neurosym as ns
import stitch_core
from permacache import permacache, stable_hash

from imperative_stitch.compress.rust_stitch.compression_result import CompressionResult
from imperative_stitch.compress.rust_stitch.process_rust_stitch import (
    process_rust_stitch,
)


@permacache(
    "imperative_stitch/compress/rust_stitch/cached_stitch_core",
    key_function=dict(s_exps=stable_hash),
)
def cached_stitch_core(
    s_exps,
    *,
    cost_prim,
    tdfa_root,
    valid_metavars,
    valid_roots,
    tdfa_non_eta_long_states,
    tdfa_split,
    **kwargs
):
    return stitch_core.compress(
        s_exps,
        cost_prim=json.dumps(cost_prim).replace(" ", ""),
        tdfa_json_path=dfa_path(),
        tdfa_root=tdfa_root,
        valid_metavars=valid_metavars,
        valid_roots=valid_roots,
        tdfa_non_eta_long_states=tdfa_non_eta_long_states,
        tdfa_split=tdfa_split,
        **kwargs,
    )


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
    compressed = cached_stitch_core(
        s_exps,
        cost_prim=cost_prim,
        tdfa_root="M",
        valid_metavars='["S","E","seqS"]',
        valid_roots='["S","E","seqS"]',
        tdfa_non_eta_long_states='{"seqS":"S"}',
        tdfa_split=ns.python_dsl.names.PYTHON_DSL_SEPARATOR,
        **kwargs,
    )
    return process_rust_stitch(compressed)


@lru_cache
def dfa_path():
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as f:
        json.dump(ns.python_dfa(), f)
    return path


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
