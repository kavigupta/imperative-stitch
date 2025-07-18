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
    return program.map(
        lambda x: (replace_fn(x) if isinstance(x, AbstractionCallAST) else x)
    )


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
    # We will keep iterating until we reach a fixed point.
    # This is necessary because the bodies may contain more abstraction calls.
    # This is a sufficient number of iterations, since each abstraction call is rendered when ruig str()
    for _ in range(len(str(program))):
        done = True

        def callback():
            nonlocal done
            done = False

        result = abstraction_calls_to_bodies(
            result, abstractions, pragmas=pragmas, callback=callback
        )
        if done:
            return result

    raise RuntimeError("Abstraction calls to bodies recursively did not converge")


def inline_multiline_calls(program, abstractions):
    """
    Inline all multiline calls in the program.
    This is a special case where we want to inline the calls that are
    multiline, i.e., they have a body that is a sequence of statements.
    """

    def mapping(node):
        if isinstance(node, AbstractionCallAST) and node.some_argument_is_multiline():
            return abstractions[node.tag].substitute_body(node.args, pragmas=False)
        return node

    return program.map(mapping)
