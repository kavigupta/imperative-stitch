import ast

from parameterized import parameterized

from imperative_stitch.analyze_program.antiunify.extract_at_multiple_sites import (
    antiunify_all_metavariables_across_extractions,
)

from imperative_stitch.analyze_program.extract.extract import do_extract
from imperative_stitch.data.parse_extract import parse_extract_pragma
from imperative_stitch.utils.ast_utils import (
    ReplaceNodes,
    ast_nodes_in_order,
    field_is_body,
)
from tests.extract_test import GenericExtractTest
from tests.rewrite_test import GenericRewriteRealisticTest
from tests.utils import canonicalize, small_set_examples


def run_extract_from_tree(tree, site):
    extr = do_extract(site, tree, extract_name="__f0")
    return extr


def run_extract(test, code, num_metavariables=None):
    try:
        code = canonicalize(code)
    except SyntaxError:
        print(code)
        test.fail("syntax error")
    tree, sites = parse_extract_pragma(code)
    code = ast.unparse(tree)
    if num_metavariables is not None:
        for site in sites:
            test.assertEqual(len(site.metavariables), num_metavariables)
    extrs = [run_extract_from_tree(tree, site) for site in sites]
    antiunify_all_metavariables_across_extractions(extrs)
    post_extracteds = {ast.unparse(extr.func_def) for extr in extrs}
    if len(post_extracteds) != 1:
        for x in post_extracteds:
            print(x)
        test.fail("not all results are the same")
    [post_extracted] = post_extracteds
    result = ast.unparse(tree), post_extracted
    for extr in extrs:
        extr.undo()
    test.assertEqual(ast.unparse(tree), code)
    return result


class AntiUnifyTest(GenericExtractTest):
    def run_extract(self, code, num_metavariables=None):
        return run_extract(self, code, num_metavariables)

    def test_basic_antiunify(self):
        code = """
        def f(x, y):
            __start_extract__
            x = x ** 3
            y = x ** 7
            return lambda u: y ** {__metavariable__, __m1, x - y} + u
            __end_extract__

        def g(x, y):
            __start_extract__
            x = x ** 3
            y = x ** 7
            return lambda v: y ** {__metavariable__, __m1, y} + v
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, lambda x, y: x - y)

        def g(x, y):
            return __f0(x, lambda __u1, y: y)
        """
        post_extracted = """
        def __f0(__0, __m1):
            __0 = __0 ** 3
            __1 = __0 ** 7
            return lambda __2: __1 ** __m1(__0, __1) + __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_no_overlap_antiunify(self):
        code = """
        def f(x, y):
            __start_extract__
            x = x ** 3
            y = x ** 7
            return lambda u: y ** {__metavariable__, __m1, x} + u
            __end_extract__

        def g(x, y):
            __start_extract__
            x = x ** 3
            y = x ** 7
            return lambda v: y ** {__metavariable__, __m1, y} + v
            __end_extract__
        """
        post_extract_expected = """
        def f(x, y):
            return __f0(x, lambda x, __u1: x)

        def g(x, y):
            return __f0(x, lambda __u1, y: y)
        """
        post_extracted = """
        def __f0(__0, __m1):
            __0 = __0 ** 3
            __1 = __0 ** 7
            return lambda __2: __1 ** __m1(__0, __1) + __2
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )

    def test_different_order(self):
        code = """
        def f():
            __start_extract__
            return [{__metavariable__, __m0, key} for (key, value) in 2]
            __end_extract__
        def g():
            __start_extract__
            return [{__metavariable__, __m0, value} for (key, value) in 2]
            __end_extract__
        """
        post_extract_expected = """
        def f():
            return __f0(lambda key, __u1: key)
        
        def g():
            return __f0(lambda __u1, value: value)
        """
        post_extracted = """
        def __f0(__m0):
            return [__m0(__0, __1) for (__0, __1) in 2]
        """
        self.assertCodes(
            self.run_extract(code), (post_extract_expected, post_extracted)
        )


class AntiUnifyRealisticTest(GenericRewriteRealisticTest):
    def run_extract(self, code, num_metavariables=None):
        return run_extract(self, code, num_metavariables)

    def sample_site(self, rng, tree):
        nodes = list(ast_nodes_in_order(tree))
        fields = [
            (n, f)
            for n in nodes
            for f in n._fields
            if field_is_body(type(n), f) and len(getattr(n, f)) > 0
        ]
        node, field = fields[rng.choice(len(fields))]
        length = len(getattr(node, field))
        start = rng.randint(length)
        end = rng.randint(start + 1, length + 1)
        body = getattr(node, field)
        body.insert(start, ast.Expr(ast.Name("__start_extract__")))
        body.insert(end + 1, ast.Expr(ast.Name("__end_extract__")))
        body[start + 1 : end + 1], count = self.manipulate(
            body[start + 1 : end + 1], rng
        )
        code = ast.unparse(tree)
        codes = [code]
        num_copies = rng.randint(1, 4)
        for _ in range(num_copies):
            try:
                tree, [site] = parse_extract_pragma(code)
            except SyntaxError:
                print(code)
                self.fail("syntax error")
            site.add_pragmas()
            for _, meta in site.metavariables:
                exprs = [
                    x
                    for x in self.get_expressions(meta, "E")
                    if ast.unparse(x) != ast.unparse(meta)
                ]
                if not exprs:
                    continue
                expr = rng.choice(exprs)
                ReplaceNodes({meta: expr}).visit(tree)
            new_code = ast.unparse(tree)
            codes.append(new_code)
        code = "\n".join(codes)
        print("=" * 80)
        print(code)
        return code, count

    @parameterized.expand([(i,) for i in range(len(small_set_examples()))])
    def test_realistic(self, i):
        self.operate(i)