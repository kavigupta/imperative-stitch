import ast
import json
import os
import re
import subprocess
from typing import Counter

import neurosym as ns
import numpy as np
import pandas as pd
import requests
import tqdm.auto as tqdm
from github import Auth, Github, UnknownObjectException
from permacache import permacache

from imperative_stitch.utils.remove_docstrings import remove_docstrings

github_link_pat = re.compile(r"(https?://github.com/([^/]+)/([^/)]+))/?")


def repos_folder():
    import imperative_stitch

    return os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.abspath(imperative_stitch.__path__[0]))
        ),
        "imperative-stitch-repos",
    )


@permacache(
    "imperative_stitch/data/github_repository_downloader/get_github_link_from_pypi_6"
)
def get_github_link_from_pypi(package):
    code = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=1000).json()
    urls = code["info"]["project_urls"]
    if not any("github" in x for x in urls.values()):
        return None
    for key_lower in [
        "source",
        "code",
        "source code",
        "repository",
        "github",
        "homepage",
        "project",
    ]:
        for key in urls:
            if key_lower == key.lower():
                mat = github_link_pat.match(urls[key])
                if mat:
                    return mat.group(1), mat.group(2), mat.group(3)
    raise ValueError(urls)


def github_api():
    # pylint: disable=consider-using-with
    auth = Auth.Token(open(os.path.expanduser("~/.github_token")).read().strip())
    return Github(auth=auth)


@permacache(
    "imperative_stitch/data/github_repository_downloader/repo_license_contents_1"
)
def repo_license_contents(repo_name):
    try:
        g = github_api()
        repo = g.get_repo(repo_name)
        license_text = repo.get_license()
        return license_text.decoded_content.decode("utf-8")
    except UnknownObjectException:
        return None


def is_open_source(repo_name):
    if repo_name in {
        "python/typing_extensions",
        "pypa/distlib",
        "fsspec/s3fs",
        "numpy/numpy",
    }:
        return True

    if repo_name in {
        "certifi/python-certifi",
        "XingangPan/DragGAN",
    }:
        return False
    license_contents = repo_license_contents(repo_name)
    if license_contents is None:
        return False

    lines = [x for x in license_contents.split("\n") if x.strip()]
    license_contents = " ".join(license_contents.split())
    if (
        "Permission is hereby granted, free of charge, to any person obtaining a copy of"
        in license_contents
        and "use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of"
        in license_contents
    ):
        return True
    if (
        "Redistribution and use in source and binary forms, with or without modification"
        in license_contents
        and "are permitted provided that the following conditions are met"
        in license_contents
    ):
        return True
    if "www.apache.org/licenses" in license_contents:
        return True
    if "Apache License, Version 2.0" in license_contents:
        return True
    if "BSD 3-Clause License" in license_contents:
        return True
    if "found in LICENSE.APACHE or LICENSE.BSD" in license_contents:
        return True
    if "licenses found in LICENSE.APACHE2 or LICENSE.MIT" in license_contents:
        return True
    if "released into the public domain" in license_contents:
        return True
    if (
        "Permission to use, copy, modify, and/or distribute this software for any"
        in license_contents
        and "purpose with or without fee is hereby granted" in license_contents
    ):
        return True
    if "HPND License" in license_contents:
        return True
    if "GNU LESSER GENERAL PUBLIC LICENSE" in license_contents:
        return True
    if "GNU GENERAL PUBLIC LICENSE" in license_contents:
        return True
    if "LLAMA 2 COMMUNITY LICENSE AGREEMENT" in license_contents:
        return True
    if (
        "Creative Commons Attribution 4.0 International License (CC BY 4.0)"
        in license_contents
    ):
        return True
    if "CREATIVE COMMONS PUBLIC LICENSE" in license_contents:
        return True
    if "Creative Commons Legal Code" in license_contents:
        return True
    if "GNU AFFERO GENERAL PUBLIC LICENSE" in license_contents:
        return True
    if "Creative Commons Attribution-NonCommercial-ShareAlike 4.0" in license_contents:
        return True
    if (
        "Everyone is permitted to copy and distribute verbatim or modified"
        in license_contents
    ):
        return True
    if "Mozilla Public License Version 2.0" in license_contents:
        return True
    raise ValueError(f"Unknown license: {repo_name} {lines[:5]}")


def get_pypi():
    # Sourced from https://hugovk.github.io/top-pypi-packages/ on 2024-06-24
    with open("data/top-pypi-packages-30-days.min.json") as f:
        top_packages = json.load(f)
    top_pypi = pd.DataFrame(top_packages["rows"])
    top_pypi = top_pypi.sort_values("download_count")[::-1]
    githubs = []
    for x in top_pypi.project:
        if x in ["pypular", "sqlalchemy", "azure-core"]:
            continue
        # print(x)
        res = get_github_link_from_pypi(x)
        # print(link)
        if res is None:
            continue
        link, first, second = res
        if not is_open_source(f"{first}/{second}"):
            continue
        githubs.append((link.strip("/"), first, second))
        if len(githubs) >= 100:
            break
    return githubs


def by_github_stars():
    # sourced from https://github.com/EvanLi/Github-Ranking/blob/master/Top100/Python.md?plain=1 on 2024-06-24
    githubs = []
    for mat in github_link_pat.finditer(
        requests.get(
            "https://raw.githubusercontent.com/EvanLi/Github-Ranking/c111d4753ec40366b03d39da66083902811a5066/Top100/Python.md",
            timeout=1000,
        ).content.decode("utf-8")
    ):
        link, first, second = mat.group(1), mat.group(2), mat.group(3)

        if not is_open_source(f"{first}/{second}"):
            continue
        githubs.append((link.strip("/"), first, second))
        if len(githubs) >= 100:
            break
    return githubs


def get_top_repos():
    github_stars = list(by_github_stars())[::-1]
    pypis = list(get_pypi())[::-1]
    result = []
    while github_stars and pypis:
        github_star = github_stars.pop()
        pypi = pypis.pop()
        result.append(github_star)
        result.append(pypi)
    result.extend(github_stars[::-1])
    result.extend(pypis[::-1])
    unique = []
    for x in result:
        if x not in unique:
            unique.append(x)
    # assert all seconds are unique
    counted = Counter(x[2] for x in unique)
    for k, v in counted.items():
        assert v == 1, [link for link, _, second in unique if second == k]
    return unique


def clone_repo(repo_link, first, second):
    try:
        os.makedirs(repos_folder())
    except FileExistsError:
        pass
    if os.path.exists(os.path.join(repos_folder(), second)):
        return
    out_path = repos_folder()
    del repo_link
    ssh_link = f"git@github.com:{first}/{second}.git"
    subprocess.check_call(["git", "clone", ssh_link], cwd=out_path)


def all_python_files(repo_link, first, second):
    """
    Get a list of all python files in a repo. Return a dictionary
    mapping the file path (relative to the repo root) to the file contents.
    """
    print(repo_link, first, second)
    clone_repo(repo_link, first, second)
    repo_path = os.path.join(repos_folder(), second)
    python_files = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            relpath = os.path.relpath(os.path.join(root, file), repo_path)
            if not file.endswith(".py"):
                continue
            if "doc" in relpath or "test" in relpath or "data" in relpath:
                continue
            try:
                with open(os.path.join(root, file), "r") as f:
                    code = f.read()
            except UnicodeDecodeError:
                continue
            try:
                ast.parse(code)
            except SyntaxError:
                continue
            python_files[relpath] = code
    return python_files


def all_repos_contents():
    path = "data/all_repos_contents/"
    if not os.path.exists(path):
        top_repos = get_top_repos()
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        for link, first, second in tqdm.tqdm(top_repos):
            k = f"{first}__{second}.json"
            v = all_python_files(link, first, second)
            with open(os.path.join(path, k), "w") as f:
                json.dump(v, f)
    result = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            result[file] = json.load(f)
    return result


@permacache(
    "imperative_stitch/data/github_repository_downloader/single_repo_random_subset_of_size"
)
def single_repo_random_subset_of_size(path, num_chars):
    """
    Randomly sample a subset of the data from the given path. The subset will
    have approximately `num_chars` characters. Docstrings are removed
    """
    with open(path, "r") as f:
        data = sorted(json.load(f).items())
    return grab_random_subset(num_chars, data)


@permacache("imperative_stitch/data/github_repository_downloader/one_each_5")
def one_each():
    """
    Get one example from each repo, randomly
    """
    rng = np.random.RandomState(0)
    data = all_repos_contents()
    result = {}
    for k, values in data.items():
        values = sorted(values.items())
        while True:
            k_sub, v = values[rng.randint(len(values))]
            try:
                ast.parse(v)
                result[k + "/" + k_sub] = v
                break
            except SyntaxError:
                pass
    return result


@permacache(
    "imperative_stitch/data/github_repository_downloader/multiple_repos_random_subset_of_size_2"
)
def multiple_repos_random_subset_of_size(num_chars):
    """
    Like `single_repo_random_subset_of_size`, but for multiple repos, only one
    example per repo is taken
    """
    data = sorted(one_each().items())
    return grab_random_subset(num_chars, data)


def grab_random_subset(num_chars, data):
    data = data.copy()
    np.random.RandomState(0).shuffle(data)
    result = []
    pbar = tqdm.tqdm(total=num_chars)
    total_chars = 0
    for k, v in data:
        v = ast.unparse(remove_docstrings(ast.parse(v)))
        ast.parse(v)
        result.append((k, ns.python_to_s_exp(v)))
        total_chars += len(v)
        pbar.update(len(v))
        if total_chars > num_chars:
            break
    else:
        raise ValueError("Not enough data")
    pbar.close()
    return dict(result)


if __name__ == "__main__":
    all_repos_contents()
