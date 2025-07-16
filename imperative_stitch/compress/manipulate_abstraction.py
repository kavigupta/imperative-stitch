from imperative_stitch.parser.python_ast import AbstractionCallAST


def collect_abstraction_calls(program):
    """
    Collect all abstraction calls in this PythonAST. Returns a dictionary
        from handle to abstraction call object.
    """
    result = {}

    def collect(x):
        if isinstance(x, AbstractionCallAST):
            result[x.handle] = x
        return x

    program.map(collect)
    return result


def replace_abstraction_calls(program, handle_to_replacement):
    """
    Replace the abstraction call with the given handle with the given replacement.
    """
    return program.map(
        lambda x: (
            handle_to_replacement.get(x.handle, x)
            if isinstance(x, AbstractionCallAST)
            else x
        )
    )


def map_abstraction_calls(program, replace_fn):
    """
    Map each abstraction call through the given function.
    """
    handle_to_replacement = collect_abstraction_calls(program)
    handle_to_replacement = {
        handle: replace_fn(call) for handle, call in handle_to_replacement.items()
    }
    return replace_abstraction_calls(program, handle_to_replacement)


def abstraction_calls_to_stubs(program, abstractions, *, is_pythonm=False):
    """
    Replace all abstraction calls with stubs. Does so via a double iteration.
        Possibly faster to use a linearization of the set of stubs.
    """
    result = program
    while True:
        abstraction_calls = collect_abstraction_calls(result)
        if not abstraction_calls:
            return result
        replacement = {}
        for handle, node in abstraction_calls.items():
            if (set(collect_abstraction_calls(node)) - {handle}) == set():
                replacement[handle] = abstractions[node.tag].create_stub(
                    node.args, is_pythonm=is_pythonm
                )
        result = replace_abstraction_calls(result, replacement)


def abstraction_calls_to_bodies(program, abstractions, *, pragmas=False, callback=None):
    """
    Replace all abstraction calls with their bodies.
    """

    def construct(call):
        if call.tag in abstractions:
            if callback is not None:
                callback()
            return abstractions[call.tag].substitute_body(call.args, pragmas=pragmas)
        return call

    return map_abstraction_calls(program, construct)


def abstraction_calls_to_bodies_recursively(program, abstractions, *, pragmas=False):
    """
    Replace all abstraction calls with their bodies, recursively.
    """
    result = program
    while True:
        done = True

        def callback():
            nonlocal done
            done = False

        result = abstraction_calls_to_bodies(
            result, abstractions, pragmas=pragmas, callback=callback
        )
        if done:
            return result
