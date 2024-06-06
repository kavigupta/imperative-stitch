import ast
import copy

from imperative_stitch.analyze_program.extract.extract import create_target


def reconfigure_parameter(parameter, variables, vars_all):
    """
    Mutate the parameter to increase the number of variables.

    E.g., if parameter is lambda x, z: x + z, and vars is [a, b], and
        vars_all is [a, b, c], then the result is lambda x, __u1, z: x + z.

    Args:
        parameter (ast.Lambda): The parameter to reconfigure.
        variables (list[str]): The current variables used to call the parameter.
        vars_all (list[str]): The eventual variables used to call the parameter.
    """
    arg_names = [x.arg for x in parameter.args.args]
    assert len(arg_names) == len(variables)
    i = 0

    def unused():
        nonlocal i
        i += 1
        return f"__u{i}"

    new_args_names = []
    for v in vars_all:
        if v in variables:
            new_args_names.append(arg_names[variables.index(v)])
        else:
            new_args_names.append(unused())
    parameter.args.args = [ast.arg(name) for name in new_args_names]


def antiunify_extractions(extrs):
    """
    Antiunify all metavariables across all extractions. Once this is done,
        all metavariables will have the same number of passed variables.

    Args:
        extrs (list[Extraction]): The extractions to antiunify.
    """
    all_metavariable_names = sorted(
        {name for extr in extrs for name in extr.metavariables.names}
    )
    for metavariable_name in all_metavariable_names:
        antiunify_metavariable_across_extractions(extrs, metavariable_name)

    antiunify_returns(extrs)

    codes = {ast.unparse(extr.func_def): extr for extr in extrs}
    if len(codes) != 1:
        print("*" * 80)
        for code in codes:
            print(code)
            print("=" * 80)
            print(ast.unparse(ast.Module(body=codes[code].call, type_ignores=[])))
            print("*" * 80)

        raise RuntimeError("not all results are the same")


def antiunify_metavariable_across_extractions(extrs, metavariable_name):
    """
    Antiunify the given metavariable across all extractions. Once this is done,
        the metavariable will have the same number of passed variables.

    Args:
        extrs (list[Extraction]): The extractions to antiunify.
        metavariable_name (str): The name of the metavariable to antiunify.
    """
    metavariables = [
        extr.metavariables.replacements_for_name(metavariable_name)[0] for extr in extrs
    ]
    parameters = [
        extr.metavariables.parameter_for_name(metavariable_name) for extr in extrs
    ]
    vars_each = []
    for meta in metavariables:
        assert meta.call.func.id == metavariable_name
        vars_each.append([arg.id for arg in meta.call.args])
    vars_all = sorted({var for variables in vars_each for var in variables})
    for extr, parameter, meta, variables in zip(
        extrs, parameters, metavariables, vars_each
    ):
        for meta in extr.metavariables.replacements_for_name(metavariable_name):
            meta.call.args = [ast.Name(var, ast.Load()) for var in vars_all]
        reconfigure_parameter(parameter, variables, vars_all)


def antiunify_returns(extrs):
    """
    Antiunify the returns across all extractions. Once this is done,
        all returns will have the same number of passed variables.

    Assumes that all the variables are named the same and
        all that needs to be changed is the return statements.
    """
    return_names_all = sorted({name for extr in extrs for name in extr.return_names})
    return_target = create_target(return_names_all, ast.Load())
    for extr in extrs:
        if extr.return_names == return_names_all:
            continue
        return_to_call = dict(zip(extr.return_names, extr.call_names))
        call_target = [return_to_call.get(name, "_") for name in return_names_all]
        if (
            extr.call_names
        ):  # if there are no call names, we don't need to change the call since the output is not used
            extr.call[0].targets[0] = create_target(call_target, ast.Store())

        for return_ in extr.returns:
            return_.value = copy.deepcopy(return_target)

        if return_names_all and (
            not extr.func_def.body or not isinstance(extr.func_def.body[-1], ast.Return)
        ):
            extr.func_def.body.append(ast.Return(value=copy.deepcopy(return_target)))
