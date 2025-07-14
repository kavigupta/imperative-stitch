import json
import os
import shlex
import subprocess
import tempfile

import appdirs

from imperative_stitch.utils.classify_nodes import export_dfa

cache_dir = appdirs.user_cache_dir("imperative_stitch")

# a, b = f(c, d)
# (Assign
#     (list (Tuple (list (Name &a:0 Store) (Name &b:0 Store)) Store))
#     (Call (Name &f:0 Load) (list (Name &c:0 Load) (Name &d:0 Load)) nil)
#     None
# )
# sizeof
#     = assign_size + n_returns * sym_size + call_size + fn_name + n_returns * sym_size
#     = assign_size + call_size + fn_name + 2 * n_returns * sym_size


def run_julia_stitch(*args, iters, **kwargs):
    output = run_julia_stitch_generic(
        "cli/compress.jl", *args, **kwargs, extra_args=[f"--iterations={iters}"]
    )
    *_, tildes0, sizes, tildes1, abstractions, tildes2, rewritten, newline = (
        output.split("\n")
    )
    assert tildes0 == tildes1 == tildes2 == "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    assert newline == ""
    sizes = json.loads(sizes)
    abstractions = json.loads(abstractions)
    rewritten = json.loads(rewritten)
    return sizes, abstractions, rewritten


def run_julia_rewrite(code, abstractions, **kwargs):
    output = run_julia_stitch_generic(
        "cli/rewrite.jl",
        code,
        **kwargs,
        extra_args=[
            f"--abstractions={json.dumps(abstractions)}",
        ],
    )
    *_, tildes, rewritten, newline = output.split("\n")
    assert tildes == "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    assert newline == ""
    rewritten = json.loads(rewritten)
    return rewritten


def run_julia_stitch_generic(
    path,
    code,
    *,
    stitch_jl_dir,
    extra_args=(),
    max_arity,
    quiet=True,
    application_utility_metavar=-1,
    application_utility_symvar=-0.5,
    application_utility_fixed=-3,  # see reasoning above
    include_dfa=True,
    root_states=("E", "S", "seqS"),
    metavariable_statements=True,
    metavariables_anywhere=False,
    minimum_number_matches=2,
):
    size_by_symbol = {
        "Module": 0.0,
        "Name": 0.0,
        "Load": 0.0,
        "Store": 0.0,
        "None": 0.0,
        "list": 0.0,
        "nil": 0.0,
        "/seq": 0.0,
        "Constant": 0.0,
        "Attribute": 0.0,
        "_slice_content": 0.0,
        "_slice_slice": 0.0,
        "_slice_tuple": 0.0,
        "_starred_content": 0.0,
        "_starred_starred": 0.0,
    }
    try:
        os.makedirs(cache_dir)
    except FileExistsError:
        pass
    dfa_file = cache_dir + "/dfa.json"
    with open(dfa_file, "w") as f:
        json.dump(export_dfa(), f, indent=2)
    code_str = json.dumps(code)

    if len(code_str) > 10**5:
        p = tempfile.mktemp()
        with open(p, "w") as f:
            f.write(code_str)
        corpus_arg = f"--corpus-file={p}"
    else:
        corpus_arg = f"--corpus={code_str}"
    cmd = [
        "julia",
        "--project=" + stitch_jl_dir,
        os.path.join(stitch_jl_dir, path),
        corpus_arg,
        f"--max-arity={max_arity}",
        *([f"--dfa={dfa_file}"] if include_dfa else []),
        f"--size-by-symbol={json.dumps(size_by_symbol)}",
        f"--application-utility-fixed={application_utility_fixed}",
        f"--application-utility-metavar={application_utility_metavar}",
        f"--application-utility-symvar={application_utility_symvar}",
        f"--dfa-valid-root-states={json.dumps(list(root_states)) if root_states is not None else 'any'}",
        *(
            []
            if metavariable_statements
            else ["--dfa-metavariable-disallow-S", "--dfa-metavariable-disallow-seqS"]
        ),
        *(["--dfa-metavariable-allow-anything"] if metavariables_anywhere else []),
        *extra_args,
        f"--minimum-number-matches={minimum_number_matches}",
    ]
    if not quiet:
        print("Run the following command to debug:")
        print(" ".join([shlex.quote(x) for x in cmd]))
    abstractions = subprocess.run(
        cmd,
        capture_output=True,
        check=False,
    ).stdout
    abstractions = abstractions.decode("utf-8")
    if not quiet:
        print(abstractions)
    return abstractions
