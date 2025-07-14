import json
from functools import lru_cache

import neurosym as ns
import tqdm
from permacache import drop_if_equal, permacache, stable_hash

from imperative_stitch.compress.julia_stitch import run_julia_stitch
from imperative_stitch.data.compression_testing_code import compression_testing_code


@permacache(
    "imperative_stitch/data/stitch_output_set/run_stitch_cached_11",
    key_function=dict(
        c=stable_hash,
        root_states=drop_if_equal(("S", "seqS")),
        iters=drop_if_equal(1),
        metavariables_anywhere=drop_if_equal(False),
    ),
)
def run_stitch_cached(
    c, root_states=("S", "seqS"), iters=1, metavariables_anywhere=False
):
    _, abstractions, rewritten = run_julia_stitch(
        c,
        stitch_jl_dir="../Stitch.jl/",
        iters=iters,
        max_arity=4,
        quiet=False,
        # TODO we should be able to root abstractions at E
        root_states=root_states,
        metavariable_statements=False,
        metavariables_anywhere=metavariables_anywhere,
    )
    return abstractions, rewritten


@permacache(
    "imperative_stitch/data/stitch_output_set/stitch_output_set_15",
    key_function=dict(
        root_states=drop_if_equal(("S", "seqS")),
        iters=drop_if_equal(1),
        metavariables_anywhere=drop_if_equal(False),
    ),
)
def stitch_output_set(
    amount, root_states=("S", "seqS"), iters=1, metavariables_anywhere=False
):
    sets = compression_testing_code(amount * 10)

    results = []

    pbar = tqdm.tqdm(total=amount)

    for datum in sets:
        s = datum["solutions"][:10]
        if len(json.dumps(s)) > 5000:
            continue
        c = [
            ns.python_to_s_exp(code, renderer_kwargs=dict(columns=float("inf")))
            for code in s
        ]

        abstractions, rewritten = run_stitch_cached(
            c,
            root_states=root_states,
            iters=iters,
            metavariables_anywhere=metavariables_anywhere,
        )

        result = dict(
            code=c,
            abstractions=abstractions,
            rewritten=rewritten,
            inputs=datum["inputs"],
            outputs=datum["outputs"],
        )

        results.append(result)
        pbar.update(1)

        if len(results) >= amount:
            pbar.close()
            break

    return results


@lru_cache(maxsize=1)
def load_stitch_output_set():
    with open("data/stitch_output_set.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_stitch_output_set_no_dfa():
    with open("data/stitch_output_set_no_dfa.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_annies_compressed_dataset():
    with open("data/annies_compressed_dataset.json") as f:
        return json.load(f)


def main():
    small = stitch_output_set(10)
    with open("data/stitch_output_set_small.json", "w") as f:
        json.dump(small, f, indent=2)

    full = stitch_output_set(100)
    with open("data/stitch_output_set.json", "w") as f:
        json.dump(full, f, indent=2)

    no_dfa = stitch_output_set(
        100, iters=20, root_states=None, metavariables_anywhere=True
    )
    with open("data/stitch_output_set_no_dfa.json", "w") as f:
        json.dump(no_dfa, f, indent=2)


if __name__ == "__main__":
    main()
