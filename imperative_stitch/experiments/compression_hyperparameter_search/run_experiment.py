from tqdm.contrib.concurrent import process_map

from imperative_stitch.experiments.compression_hyperparameter_search.hyperparameters import (
    sample_hyperparameters,
)

from .datasets import datasets
from .run_stitch import run_stitch_with_hyperparameters


def run_experiment_up_to_seed(num_seeds, hypers_for_seed, *, skip_missing):
    data = datasets()

    arguments = [
        (k, seed, hypers_for_seed(seed), skip_missing)
        for seed in range(num_seeds)
        for k in data
    ]
    results_flat = process_map(run_experiment, arguments, max_workers=32)
    results_flat = {
        (k, seed): result for (k, seed, _, _), result in zip(arguments, results_flat)
    }

    results = []
    for seed in range(num_seeds):
        result = {}
        for k in data:
            result[k] = results_flat[k, seed]
        results.append(result)
    return results


def run_experiment(k_seed_hypers_skip_missing):
    k, seed, hypers, skip_missing = k_seed_hypers_skip_missing
    if k == ("ml_repo", 1000000):
        return None
    params = dict(iters=10)
    params.update(hypers)
    data = datasets()
    kwargs = dict(
        dataset=[v for _, v in sorted(data[k].items())],
        stitch_jl_dir="../Stitch.jl/",
        **params,
        print_before_after=f"{k} with seed {seed}",
    )
    if skip_missing and not run_stitch_with_hyperparameters.cache_contains(**kwargs):
        print(f"Skipping {k} with seed {seed}")
        return None
    # pylint: disable=missing-kwoa
    return run_stitch_with_hyperparameters(**kwargs)


def vary_min_num_matches():
    minimum_number_matches = [2, 3, 4, 5, 10, 20]

    def hypers(seed):
        return {
            "max_arity": 3,
            "minimum_number_matches": minimum_number_matches[seed],
            "application_utility_metavar": -1.5,
            "application_utility_symvar": -0.2,
            "application_utility_fixed": -3,
        }

    return len(minimum_number_matches), hypers


def with_only_2_min_matches(seed):
    h = sample_hyperparameters(seed)
    h["minimum_number_matches"] = 2
    return h


def with_only_2_min_matches_and_some_auf(seed):
    h = with_only_2_min_matches(seed)
    # ensure that AUF is at most -1
    h["application_utility_fixed"] += -1
    # more iterations
    h["iters"] = 100
    return h


if __name__ == "__main__":
    run_experiment_up_to_seed(
        100, with_only_2_min_matches_and_some_auf, skip_missing=False
    )
