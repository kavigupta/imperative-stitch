from imperative_stitch.data.github_repository_downloader import (
    multiple_repos_random_subset_of_size,
    single_repo_random_subset_of_size,
)

ml_repo = "data/all_repos_contents/huggingface__transformers.json"
system_repo = "data/all_repos_contents/ray-project__ray.json"

sizes = [10_000, 30_000, 100_000, 300_000, 1_000_000]


def datasets_for_size(size):
    return {
        "ml_repo": single_repo_random_subset_of_size(ml_repo, size),
        "system_repo": single_repo_random_subset_of_size(system_repo, size),
        "across_repos": multiple_repos_random_subset_of_size(size),
    }


def datasets():
    return {
        (name, size): dataset
        for size in sizes
        for name, dataset in datasets_for_size(size).items()
    }
