from prm_eval_utils import *
from accelerate import Accelerator

from bon_eval_utils import eval_gsm8k, eval_math_prm
from math_utils.eval_theoremqa import eval_theoremqa
from math_utils import evaluate_math

import pandas as pd
from prepare import process_single_data

accelerator = Accelerator()
bon_dataset="math"
file_list, origin_dataset = get_raw_data(bon_dataset)
for file_index, file_name in enumerate(file_list):
    query_only_dataset = []
    with_output_dataset = []
    prompt_list = []
    print(file_name)
    queries = load_data(file_name, origin_dataset, accelerator)
    prompt_idx = -1

    eval_fn = {"math":evaluate_math, "gsm":eval_gsm8k, "qa":eval_theoremqa}
    for query in queries:

        ground_truth = query["answer"] if "answer" in query.keys() else query["solution"]
        if bon_dataset == "math":
            ground_truth = process_single_data(query["prompt"], ground_truth)
        if query["prompt"] not in prompt_list:
            prompt_list.append(query["prompt"])
            query_only_dataset.append({
                "task": "math",
                "idx": prompt_list.index(query["prompt"]),
                "prompt": query["prompt"],
                "reference": ground_truth,
            })
        eval_results = eval_fn[bon_dataset](query["response"], str(ground_truth), ground_truth) \
                            if bon_dataset=="qa" else \
                            eval_fn[bon_dataset](query["response"], ground_truth)
        correctness = eval_results[0]
        extracted_output = eval_results[-1].replace(" ", "")
        with_output_dataset.append({
            "task": "math",
            "idx": prompt_list.index(query["prompt"]),
            "prompt": query["prompt"],
            "response": query["response"],
            "extracted_output": extracted_output,
            "reference": str(query["answer"]) if "answer" in query.keys() else str(query["solution"]),
            "correctness": correctness
        })
    pd.DataFrame(query_only_dataset).to_json(f"/home/test/test05/ylf/prm_eval/testset_prompt/{file_name.split('/')[-1]}.json", orient="records", indent=4)
    pd.DataFrame(with_output_dataset).to_json(f"/home/test/test05/ylf/prm_eval/testset/{file_name.split('/')[-1]}.json", orient="records", indent=4)
    