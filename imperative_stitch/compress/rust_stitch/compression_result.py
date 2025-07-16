from dataclasses import dataclass
from typing import List

import neurosym as ns

from imperative_stitch.compress.abstraction import Abstraction
from imperative_stitch.compress.manipulate_abstraction import (
    abstraction_calls_to_bodies_recursively,
    abstraction_calls_to_stubs,
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

    def inline_abstractions(self, *, abstraction_names):
        """
        Inline the abstractions in the rewritten code and remaining abstractions.
        """
        abstr_dict = self.abstr_dict
        abstr_dict = {name: abstr_dict[name] for name in abstraction_names}
        new_abstractions = [x for x in self.abstractions if x.name not in abstr_dict]
        new_rewritten = [
            abstraction_calls_to_bodies_recursively(program, abstr_dict)
            for program in self.rewritten
        ]
        new_abstractions = [
            abstr.map_body(
                lambda x: abstraction_calls_to_bodies_recursively(x, abstr_dict)
            )
            for abstr in new_abstractions
        ]
        return CompressionResult(
            abstractions=new_abstractions,
            rewritten=new_rewritten,
        )
