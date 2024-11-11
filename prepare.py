# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import re
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
import pandas as pd

from math_utils.normalizer import normalize_answer_string

from math_utils.grader import extract_answer


DOWNLOAD_LINK = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"


def extract_attributes_from_name(file_name):
    """Extract attributes from file path."""
    eval_set, problem_type, fileid = file_name.split("/")[1:]
    fileid = fileid.split(".")[0]
    return eval_set, problem_type, fileid


def extract_answer_string_2(answer_str):
    """For two cases, inside the boxed expression, we needed a second iteration of parsing."""
    left_string = "\\boxed"
    idx = answer_str.rfind(left_string)

    stripped_answer = answer_str[idx + len(left_string) :]
    right_idx = stripped_answer.rfind("$")

    stripped_answer = stripped_answer[:right_idx]
    return stripped_answer


def _post_fix(problem_id, soln_string):
    """Post fixing some answer strings"""
    if problem_id == "test/intermediate_algebra/78.json":
        soln_string = re.sub(r"\\(\d+)", r"\1", soln_string)

    return soln_string


def process_single_data(question, reference_solution):
    answer_string = extract_answer(reference_solution)

    if answer_string is None:
        answer_string = extract_answer_string_2(reference_solution)

    parsed_answer = normalize_answer_string(answer_string)
    if not (
        ("Find the equation" in question)
        or ("Enter the equation" in question)
        or ("What is the equation") in question
        or ("described by the equation") in question
        or ("Find an equation") in question
    ) and ("=" in parsed_answer):
        if parsed_answer.count("=") == 1:
            # For greater count, it means we're just predicting values of multiple variables
            parsed_answer = parsed_answer.split("=")[1]
    
    return parsed_answer

def process_data():
    """Download tar and condense data into single jsonl file."""
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--split_name",
    #     required=True,
    #     choices=("test", "validation", "train", "train_full"),
    # )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=1000)
    # parser.add_argument("--prompt_type", default="code_sfted")
    args = parser.parse_args()

    output_folder = Path(__file__).absolute().parent
    output_folder.mkdir(exist_ok=True)
    # actual_split_name = "test" if args.split_name == "test" else "train"

    # with tempfile.TemporaryDirectory() as temp_dir:
    #     archive_filename = os.path.join(temp_dir, "temp.tar")
    #     urllib.request.urlretrieve(DOWNLOAD_LINK, archive_filename)

    #     split_instances_dict = defaultdict(list)

    #     with tarfile.TarFile(archive_filename, mode="r") as reader_f:
    #         for tar_member in reader_f:
    #             filename = tar_member.name
    #             if not filename.endswith(".json"):
    #                 continue

    #             eval_set, problem_type, fileid = extract_attributes_from_name(filename)
    #             # TODO: we should just process all at ones, not do duplicate computation
    #             if eval_set != actual_split_name:
    #                 continue

    #             content = json.loads(reader_f.extractfile(tar_member).read())
    new_test = []
    all_problems = pd.read_json(path).to_dict(orient="records")
    for id, content in enumerate(all_problems):
            content["question"] = content["prompt"]
            content["reference_solution"] = content["reference"]

            
            content["answer"] = process_single_data(content["question"], content["reference_solution"])

            # Sanity check that content type matches the parent folder
            # content_type = content["type"].lower()
            # content_type = content_type.replace(" ", "_")
            # content_type = content_type.replace("&", "and")
            # assert problem_type == content_type

            # content["id"] = f"test/{content_type}/{id}.json"
            # content["answer"] = _post_fix(content["id"], content["answer"])
            del content["question"]
            del content["reference_solution"]
            new_test.append(content)
    print(new_test[0])
    df = pd.DataFrame(new_test)
    df.to_json(path, orient="records", indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    process_data(args.path)