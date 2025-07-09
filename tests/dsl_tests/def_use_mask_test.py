import copy
import unittest

import neurosym as ns

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies,
    abstraction_calls_to_bodies_recursively,
    abstraction_calls_to_stubs,
)
from imperative_stitch.data.stitch_output_set import (
    load_stitch_output_set,
    load_stitch_output_set_no_dfa,
)
from imperative_stitch.parser import converter
from imperative_stitch.utils.classify_nodes import export_dfa
from tests.dsl_tests.utils import fit_to
from tests.utils import (
    cwq,
    expand_with_slow_tests,
    load_annies_compressed_individual_programs,
    parse_with_hijacking,
)


class DefUseMaskTestGeneric(unittest.TestCase):
    def annotate_alternates(self, chosen, alts):
        self.assertIn(chosen, alts)
        mat = ns.python_def_use_mask.match_either_name_or_global(chosen)
        if not mat:
            return chosen
        name, scope = mat.group("name"), (
            mat.group("scope") if mat.group("typ") == "&" else "0"
        )
        # print(alts)
        alts = [ns.python_def_use_mask.match_either_name_or_global(alt) for alt in alts]
        # print([x for x in alts if x])
        alts = {x.group("name") for x in alts if x}
        alts.remove(name)
        alts = sorted(alts)
        if alts:
            name = f"{name}?{'$'.join(alts)}"
        return f"const-&{name}:{scope}~Name"

    def annotate_program(
        self,
        program,
        parser=ns.python_to_python_ast,
        abstrs=(),
        convert_to_python=True,
    ):
        dfa, _, fam, _ = fit_to(
            [program], parser=parser, abstrs=abstrs, include_type_preorder_mask=False
        )
        annotated = converter.s_exp_to_python_ast(
            ns.render_s_expression(
                ns.annotate_with_alternate_symbols(
                    ns.to_type_annotated_ns_s_exp(parser(program), dfa, "M"),
                    fam.tree_distribution_skeleton,
                    self.annotate_alternates,
                )
            )
        )
        if convert_to_python:
            annotated = abstraction_calls_to_stubs(
                annotated,
                {x.name: x for x in abstrs},
            )
            return annotated.to_python()
        return annotated

    def assertAbstractionAnnotation(
        self, code, rewritten, abstractions, convert_to_python=True
    ):
        print("*" * 80)
        for abstr in abstractions:
            print(ns.render_s_expression(abstr.body.to_ns_s_exp()))
        print("*" * 80)
        print(
            abstraction_calls_to_stubs(
                converter.s_exp_to_python_ast(code),
                {x.name: x for x in abstractions},
            ).to_python()
        )
        print("*" * 80)
        if convert_to_python:
            print(
                abstraction_calls_to_stubs(
                    converter.s_exp_to_python_ast(rewritten),
                    {x.name: x for x in abstractions},
                ).to_python()
            )
        print("*" * 80)
        try:
            self.annotate_program(
                code,
                parser=converter.s_exp_to_python_ast,
                abstrs=abstractions,
            )
        except AssertionError:
            return
        self.annotate_program(
            rewritten,
            parser=converter.s_exp_to_python_ast,
            abstrs=abstractions,
            convert_to_python=convert_to_python,
        )


class DefUseMaskWithAbstractionsTest(DefUseMaskTestGeneric):
    abstr_two_assigns = Abstraction.of(
        "fn_1",
        "(/seq (Assign (list (Name %2 Store)) (Name %1 Load) None) (Assign (list (Name %2 Store)) #0 None))",
        "seqS",
        dfa_symvars=["Name"] * 2,
        dfa_metavars=["E"],
    )

    def blank_abstraction(self, name, content):
        return Abstraction.of(name, ns.python_statements_to_python_ast(content), "seqS")

    def test_with_empty_abstraction(self):
        code = cwq(
            """
            u = 2
            k = u.count()
            "~(/splice (fn_1))"
            k = u.count()
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[self.blank_abstraction("fn_1", "a = int(input()); z = input()")],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                u?a$input$int$k$z = 2
                k?a$input$int$u$z = u?input$int.count()
                fn_1()
                k?a$input$int$u$z = u?a$input$int$k$z.count()
                """
            ).strip(),
        )

    def test_with_empty_abstraction_multi(self):
        code = cwq(
            """
            u = 2
            k = u.count()
            "~(/splice (fn_1))"
            k = u.count()
            "~(/splice (fn_2))"
            k = u.count()
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                self.blank_abstraction("fn_1", "a = int(input()); z = input()"),
                self.blank_abstraction("fn_2", "x = 3"),
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                u?a$input$int$k$x$z = 2
                k?a$input$int$u$x$z = u?input$int.count()
                fn_1()
                k?a$input$int$u$x$z = u?a$input$int$k$z.count()
                fn_2()
                k?a$input$int$u$x$z = u?a$input$int$k$x$z.count()
                """
            ).strip(),
        )

    def test_with_symvars_ordered(self):
        code = cwq(
            """
            b = 2
            "~(fn_1 &a:0 &b:0)"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(Assign (list (Name %1 Store)) (Name %2 Load) None)",
                    "S",
                    dfa_symvars=["Name"] * 2,
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                fn_1(__ref__(a?b), __ref__(b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_with_symvars_backwards(self):
        code = cwq(
            """
            b = 2
            "~(fn_1 &b:0 &a:0)"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(Assign (list (Name %2 Store)) (Name %1 Load) None)",
                    "S",
                    dfa_symvars=["Name"] * 2,
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                fn_1(__ref__(b), __ref__(a?b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_with_metavariable_very_simple(self):
        code = cwq(
            """
            b = 2
            "~(fn_1 (Name &b:0 Load) &a:0)"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(Assign (list (Name %1 Store)) #0 None)",
                    "S",
                    dfa_symvars=["Name"],
                    dfa_metavars=["E"],
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                fn_1(__code__('b'), __ref__(a?b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_symvar_reuse(self):
        code = cwq(
            """
            b = 2
            "~(/splice (fn_1 &b:0 &a:0))"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(/seq (Assign (list (Name %2 Store)) (Name %1 Load) None) (Assign (list (Name %2 Store)) (Name %1 Load) None))",
                    "seqS",
                    dfa_symvars=["Name"] * 2,
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                fn_1(__ref__(b), __ref__(a?b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_symvar_used_in_metavariable(self):
        code = cwq(
            """
            b = 2
            "~(/splice (fn_1 (Name &a:0 Load) &b:0 &a:0))"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[self.abstr_two_assigns],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                # a?b = a?b
                fn_1(__code__('a?b'), __ref__(b), __ref__(a?b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_definition_in_metavar(self):
        code = cwq(
            """
            b = 2
            "~(/splice (fn_1 (Assign (list (Name &c:0 Store)) (Name &b:0 Load) None) &c:0 &a:0))"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(/seq #0 (Assign (list (Name %2 Store)) (Name %1 Load) None))",
                    "seqS",
                    dfa_symvars=["Name"] * 2,
                    dfa_metavars=["S"],
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a$c = 2
                # c?a$b = b
                # a?b$c = c?b
                fn_1(__code__('c?a$b = b'), __ref__(c?b), __ref__(a?b$c))
                a?b$c = a?b$c
                """
            ).strip(),
        )

    def test_definition_in_choiceseq(self):
        code = cwq(
            """
            b = 2
            "~(/splice (fn_1 &c:0 &a:0 (/choiceseq (Assign (list (Name &c:0 Store)) (Name &b:0 Load) None))))"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of(
                    "fn_1",
                    "(/seq ?0 (Assign (list (Name %2 Store)) (Name %1 Load) None))",
                    "seqS",
                    dfa_symvars=["Name"] * 2,
                    dfa_choicevars=["seqS"],
                )
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a$c = 2
                # c?a$b = b
                # a?b$c = c?b
                fn_1(__ref__(c?b), __ref__(a?b$c), __code__('c?a$b = b'))
                a?b$c = a?b$c
                """
            ).strip(),
        )

    def test_impossible_likelihood_bug(self):
        # ensure that the likelihood computation doesn't crash
        # see https://github.com/kavigupta/neurosym-lib/pull/78 which fixed this
        code = cwq(
            """
            "~(/splice (fn_1 &b:0 &a:0 (/choiceseq)))"
            a = a
            """
        )
        program = parse_with_hijacking(code)
        abstrs = [
            Abstraction.of(
                "fn_1",
                "(/seq (Assign (list (Name %2 Store)) (Name %1 Load) None) ?0)",
                "seqS",
                dfa_symvars=["Name"] * 2,
                dfa_choicevars=["seqS"],
            )
        ]
        dfa, _, fam, dist = fit_to([program], parser=lambda x: x, abstrs=abstrs)
        print(program)
        print(dist)
        self.assertEqual(
            fam.compute_likelihood(
                dist,
                ns.to_type_annotated_ns_s_exp(program, dfa, "M"),
            ),
            -float("inf"),
        )

    def test_out_of_order_within_abstraction(self):
        fn_3 = Abstraction.of(
            "fn_3",
            """
            (Expr
                (ListComp
                    #0
                    (list
                        (comprehension
                            (Name %1 Store)
                            (Call
                                #1
                                (list (_starred_content (Constant i10 None)))
                                nil)
                            nil
                            i0))))
            """,
            "S",
            dfa_symvars=["Name"],
            dfa_metavars=["E", "E"],
        )
        code = cwq(
            """
            "~(fn_3 (Name &x:0 Load) (Constant i2 None) &x:0)"
            x = 2
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[fn_3],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                fn_3(__code__('x'), __code__('2'), __ref__(x))
                x = 2
                """
            ).strip(),
        )

    def test_metavar_containing_abstraction(self):
        code = cwq(
            """
            b = 2
            "~(fn_2 (fn_1) &a:0)"
            a = a
            """
        )
        annotated = self.annotate_program(
            code,
            parser=parse_with_hijacking,
            abstrs=[
                Abstraction.of("fn_1", "(Name &b:0 Load)", "E"),
                Abstraction.of(
                    "fn_2",
                    "(Assign (list (Name %1 Store)) #0 None)",
                    "S",
                    dfa_symvars=["Name"],
                    dfa_metavars=["E"],
                ),
            ],
        )
        print(annotated)
        self.assertEqual(
            cwq(annotated).strip(),
            cwq(
                """
                b?a = 2
                # a?b = b
                fn_2(__code__('fn_1()'), __ref__(a?b))
                a?b = a?b
                """
            ).strip(),
        )

    def test_targets_containing_abstraction(self):
        self.maxDiff = None
        code = converter.s_exp_to_python_ast(
            """
            (Module~M
                (/seq~seqS~2
                    (Assign~S
                        (fn_1)
                        (Tuple~E
                            (list~_StarredRoot_~2
                                (_starred_content~StarredRoot (Constant~E (const-i2~Const) (const-None~ConstKind)))
                                (_starred_content~StarredRoot (Constant~E (const-i3~Const) (const-None~ConstKind))))
                            (Load~Ctx))
                        (const-None~TC))
                    (Assign~S
                        (list~_L_~1 (Name~L (const-&x:0~Name) (Store~Ctx)))
                        (Name~E (const-&a:0~Name) (Load~Ctx)) (const-None~TC)))
                (list~_TI_~0))
            """
        )

        abstrs = [
            Abstraction.of(
                "fn_1",
                """
                (list~_L_~1
                    (Tuple~L
                        (list~_L_~2
                            (_starred_content~L (Name~L (const-&a:0~Name) (Store~Ctx)))
                            (_starred_content~L (Name~L (const-&b:0~Name) (Store~Ctx))))
                        (Store~Ctx)))
                """,
                "[L]",
            )
        ]
        annotated = self.annotate_program(
            code,
            parser=lambda x: x,
            abstrs=abstrs,
            convert_to_python=False,
        )
        expected = """
        (Module~M
            (/seq~seqS~2
                (Assign~S
                    (fn_1~_L_)
                    (Tuple~E
                        (list~_StarredRoot_~2
                            (_starred_content~StarredRoot (Constant~E (const-i2~Const) (const-None~ConstKind)))
                            (_starred_content~StarredRoot (Constant~E (const-i3~Const) (const-None~ConstKind))))
                        (Load~Ctx))
                    (const-None~TC))
                (Assign~S
                    (list~_L_~1 (Name~L (const-&x?a$b:0~Name) (Store~Ctx)))
                    (Name~E (const-&a?b:0~Name) (Load~Ctx)) (const-None~TC)))
            (list~_TI_~0))
        """
        expected = ns.render_s_expression(ns.parse_s_expression(expected))
        self.assertEqual(
            ns.render_s_expression(
                ns.to_type_annotated_ns_s_exp(annotated, export_dfa(), "M")
            ),
            expected,
        )


class DefUseMaskWithAbstractionsRealisticTest(DefUseMaskTestGeneric):
    def check_use_mask(self, x, **kwargs):
        x = copy.deepcopy(x)
        abstractions = [
            Abstraction.of(name=f"fn_{it + 1}", **abstr)
            for it, abstr in enumerate(x["abstractions"])
        ]
        for code, rewritten in zip(x["code"], x["rewritten"]):
            self.assertAbstractionAnnotation(code, rewritten, abstractions, **kwargs)

    @expand_with_slow_tests(len(load_stitch_output_set()), 10)
    def test_realistic_with_abstractions(self, i):
        self.check_use_mask(load_stitch_output_set()[i])

    @expand_with_slow_tests(len(load_stitch_output_set_no_dfa()), 10)
    def test_realistic_with_abstractions_no_dfa(self, i):
        self.check_use_mask(load_stitch_output_set_no_dfa()[i], convert_to_python=False)


class DefUseMaskWithAbstractionsRealisticAnnieSetTest(DefUseMaskTestGeneric):
    @expand_with_slow_tests(len(load_annies_compressed_individual_programs()), 10)
    def test_annies_compressed_with_abstractions(self, i):
        abstrs, rewritten = load_annies_compressed_individual_programs()[i]
        abstrs_dict = {x.name: x for x in abstrs}

        code = ns.render_s_expression(
            abstraction_calls_to_bodies_recursively(
                converter.s_exp_to_python_ast(rewritten), abstrs_dict
            ).to_ns_s_exp()
        )
        self.assertAbstractionAnnotation(code, rewritten, abstrs)


class DefUseMaskWihAbstractionsLikliehoodAnnieSetTest(DefUseMaskTestGeneric):
    @expand_with_slow_tests(len(load_annies_compressed_individual_programs()), 10)
    def test_annies_compressed_realistic(self, i):
        abstrs, rewritten = load_annies_compressed_individual_programs()[i]
        rewritten = converter.s_exp_to_python_ast(rewritten)
        code = abstraction_calls_to_bodies(rewritten, {x.name: x for x in abstrs})
        dfa, _, fam, dist = fit_to([rewritten, code], parser=lambda x: x, abstrs=abstrs)
        # should not error
        fam.compute_likelihood(
            dist,
            ns.to_type_annotated_ns_s_exp(rewritten, dfa, "M"),
        )
