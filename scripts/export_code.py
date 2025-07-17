import json
from imperative_stitch.compress.rust_stitch import convert_all_to_annotated_s_exps
from tests.utils import small_set_examples


for count in 10, 100, 1000:
    with open(f"../stitch/data/python/{count}.json", "w") as f:
        json.dump(
            convert_all_to_annotated_s_exps(small_set_examples()[:: 1000 // count])[0],
            f,
            indent=2,
        )
