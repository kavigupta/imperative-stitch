import time

from permacache import drop_if, permacache, stable_hash

from imperative_stitch.compress.julia_stitch import run_julia_stitch


@permacache(
    "imperative_stitch/experiments/compression_hyperparameter_search/run_stitch_with_hyperparameters",
    key_function=dict(
        dataset=stable_hash,
        stitch_jl_dir=None,
        print_before_after=drop_if(lambda _: True),
    ),
    multiprocess_safe=True,
)
def run_stitch_with_hyperparameters(
    dataset,
    stitch_jl_dir,
    *,
    iters,
    max_arity,
    minimum_number_matches,
    application_utility_metavar,
    application_utility_symvar,
    application_utility_fixed,
    root_states=("S", "seqS"),
    metavariable_statements=False,
    metavariables_anywhere=False,
    current_stitch_version="configurable-minimal-number-matches-7a5b4a7",
    print_before_after=None,
):
    """
    Run stitch with the given hyperparameters.

    :param dataset: The dataset to run stitch on, must be a list of strings.
    :param stitch_jl_dir: The directory containing the stitch Julia code. Not part of the key because this
        varies without changing the result. It is up to the caller to ensure that the correct directory is
        passed and that it matches `current_stitch_version`.
    :param current_stitch_version: The version of stitch to use. This is part of the key because the results
        may vary between versions.
    :param <hyperparameter>: The hyperparameters to use for stitch.
    """
    if print_before_after is not None:
        print(f"Starting {print_before_after}")
    assert current_stitch_version == "configurable-minimal-number-matches-7a5b4a7"
    wall_time_start = time.time()
    result = run_julia_stitch(
        dataset,
        stitch_jl_dir=stitch_jl_dir,
        iters=iters,
        max_arity=max_arity,
        quiet=True,
        root_states=root_states,
        metavariable_statements=metavariable_statements,
        metavariables_anywhere=metavariables_anywhere,
        minimum_number_matches=minimum_number_matches,
        application_utility_metavar=application_utility_metavar,
        application_utility_symvar=application_utility_symvar,
        application_utility_fixed=application_utility_fixed,
    )
    wall_time_end = time.time()
    wall_time = wall_time_end - wall_time_start
    if print_before_after is not None:
        print(f"Finished {print_before_after} in {wall_time:.2f} seconds")
    return result, wall_time
