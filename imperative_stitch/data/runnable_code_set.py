import ast
import itertools
import json

import tqdm.auto as tqdm

from permacache import permacache
from datasets import load_dataset

from imperative_stitch.utils.run_code import passes_tests


def wrap(code):
    body = ast.parse(code).body
    imports = []
    for node in body:
        if isinstance(node, ast.Import):
            imports.append(node)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node)
        else:
            break
    body = body[len(imports) :]
    return ast.unparse(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    *imports,
                    ast.FunctionDef(
                        name="_main",
                        args=[],
                        body=body,
                        decorator_list=[],
                    ),
                    ast.Expr(
                        ast.Call(
                            func=ast.Name(id="_main", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
                type_ignores=[],
            )
        )
    )


def extract_from_data(datapoint, *, max_tests, max_solutions):
    """
    Extract runnable code from a datapoint.

    Returns a list of dicts with keys:
        name: str
        inputs: list[str]
        outputs: list[str]
        solution: str
    """
    name = datapoint["name"]
    tests = (
        datapoint["public_tests"],
        datapoint["private_tests"],
        datapoint["generated_tests"],
    )
    inputs, outputs = [], []
    for test in tests:
        inputs += test["input"]
        outputs += test["output"]
    inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    solutions = [
        sol
        for lang, sol in zip(
            datapoint["solutions"]["language"], datapoint["solutions"]["solution"]
        )
        if lang == 3  # Python3
    ]
    solutions = solutions[:max_solutions]
    for i, sol in enumerate(tqdm.tqdm(solutions)):
        sol = wrap(sol)
        if not passes_tests(sol, inputs, outputs):
            continue
        yield dict(name=f"{name}_{i}", inputs=inputs, outputs=outputs, solution=sol)


@permacache(
    "imperative_stitch/data/runnable_code_set/runnable_code_dataset_3",
)
def runnable_code_dataset(
    *, amount, max_solutions_per_datapoint, max_tests_per_datapoint
):
    """
    Load the runnable code dataset.

    Args:
        max_solutions_per_datapoint: int
            The maximum number of solutions to extract from each datapoint.
        max_tests_per_datapoint: int
            The maximum number of tests to include for each datapoint.

    Returns a list of dicts with keys:
        name: str
        inputs: list[str]
        outputs: list[str]
        solution: str
    """
    dataset = load_dataset("deepmind/code_contests", split="train")
    result = []
    for i in itertools.count():
        data = dataset[i]
        data = list(
            extract_from_data(
                data,
                max_solutions=max_solutions_per_datapoint,
                max_tests=max_tests_per_datapoint,
            )
        )
        result += data
        print(len(result))
        if len(result) >= amount:
            break
    return result[:amount]


if __name__ == "__main__":
    with open("data/small_set_runnable_code.json", "w") as f:
        json.dump(
            runnable_code_dataset(
                amount=10_000,
                max_solutions_per_datapoint=10,
                max_tests_per_datapoint=10,
            ),
            f,
            indent=2,
        )