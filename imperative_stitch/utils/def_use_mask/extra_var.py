import re
from dataclasses import dataclass

from imperative_stitch.utils.types import SEPARATOR


@dataclass(frozen=True, eq=True, order=True)
class ExtraVar:
    """
    Used to represent an extra variable, not found in the tree distribution.

    Used in handling De Bruijn variables.
    """

    id: int

    @classmethod
    def from_name(cls, name):
        mat = canonicalized_python_name_leaf_regex.match(name)
        if mat:
            return cls(int(mat.group("var")))
        return None

    def leaf_name(self):
        return canonicalized_python_name_as_leaf(self.id, "Name")


canonicalized_python_name_leaf_regex = re.compile(
    r"const-&(__(?P<var>\d+)):[0-9]+(" + re.escape(SEPARATOR) + r"[A-Za-z])?"
)


def canonicalized_python_name_as_leaf(name, use_type=False):
    """
    Get the canonicalized python name as a leaf node. E.g., __0
    """
    result = f"const-&{canonicalized_python_name(name)}:0"
    if use_type:
        # This is a bit of a hack, since we should really be using use_type
        # however, this would require us to add a leaf for every version of
        # use_type to the tree distribution
        result += SEPARATOR + "Name"
    return result


def canonicalized_python_name(name):
    return f"__{name}"