from dataclasses import dataclass
from typing import List

import tqdm

import neurosym as ns

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies_recursively,
    abstraction_calls_to_stubs,
    inline_multiline_calls,
)


@dataclass
class CompressionResult:
    abstractions: list[Abstraction]
    rewritten: list[ns.PythonAST]

    @property
    def abstr_dict(self):
        """
        Returns a dictionary mapping abstraction names to Abstraction objects.
        """
        return {abstr.name: abstr for abstr in self.abstractions}

    def abstractions_python(self, *, is_pythonm=False) -> List[str]:
        """
        Returns the abstractions in Python format.
        """
        return [
            abstraction_calls_to_stubs(
                x.body_with_variable_names(), self.abstr_dict, is_pythonm=is_pythonm
            ).to_python()
            for x in self.abstractions
        ]

    def rewritten_python(self, *, is_pythonm=False) -> List[str]:
        """
        Returns the rewritten code in Python format.
        """
        return [
            abstraction_calls_to_stubs(
                x, self.abstr_dict, is_pythonm=is_pythonm
            ).to_python()
            for x in self.rewritten
        ]

    def map_programs(self, fn):
        new_rewritten = [fn(program) for program in self.rewritten]
        new_abstractions = [abstr.map_body(fn) for abstr in self.abstractions]
        return CompressionResult(
            abstractions=new_abstractions,
            rewritten=new_rewritten,
        )

    def inline_abstractions(self, *, abstraction_names):
        """
        Inline the abstractions in the rewritten code and remaining abstractions.
        """
        abstr_dict = self.abstr_dict
        abstr_dict = {name: abstr_dict[name] for name in abstraction_names}
        new_abstractions = [x for x in self.abstractions if x.name not in abstr_dict]
        return CompressionResult(new_abstractions, self.rewritten).map_programs(
            lambda x: abstraction_calls_to_bodies_recursively(x, abstr_dict)
        )

    def remove_unhelpful_abstractions(self, *, is_pythonm, cost_fn):
        """
        Remove abstractions that do not help in reducing the cost.
        """
        current = self
        cost = sum(cost_fn(x) for x in current.rewritten_python(is_pythonm=is_pythonm))
        for abstr in tqdm.tqdm(
            current.abstractions, desc="Removing unhelpful abstractions"
        ):
            new = current.inline_abstractions(abstraction_names=[abstr.name])
            if is_pythonm:
                new = new.inline_multiline_calls()
            new_cost = sum(
                cost_fn(x) for x in new.rewritten_python(is_pythonm=is_pythonm)
            )
            if new_cost < cost:
                print(f"Removed {abstr.name}, saved {cost - new_cost} tokens.")
                current = new
                cost = new_cost
        return current

    def inline_multiline_calls(self):
        """
        Inline multiline calls in the rewritten code.
        """
        abstractions = self.abstr_dict
        return self.map_programs(lambda x: inline_multiline_calls(x, abstractions))
