import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import re
import argparse
import pandas as pd
import json
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
import ray
import contextlib
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
from prm_eval_utils import *



def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # if hasattr(torch._C, '_cuda_resetAccumulatedMemoryStats'):
    #     torch._C._cuda_resetAccumulatedMemoryStats()
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    # This next line is speculative and may not exist or work as intended
    if hasattr(torch._C, '_cuda_resetInBadFork'):
        torch._C._cuda_resetInBadFork()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    ray.shutdown()
    multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("cleaned up!")



def load_model(model_type):

    path = {
        "llama-3.1-8b-inst": "/home/test/test05/ylf/models/llama3.1-8b-instruct",
        "llama-3.1-8b-base": "/home/test/test05/ylf/models/llama-3.1-8b",
        # "eurus-7b-kto": "/home/test/test05/ylf/models/eurus-7b-kto",
        "llama-3-8b-eurus": "/home/test/test05/ylf/checkpoints/llama3-os-ultrafeedback+ultrainteract-pair-v2-opensource-beta_0.1-chosen_ratio_1.33-20240927043135-1-hf",
        "llama-3.1-70b-inst": "/home/test/test05/ylf/models/llama-3.1-70b-instruct",
        "qwen-2.5-72b-inst": "/home/test/test05/ylf/models/Qwen2.5-72B-Instruct"
    }
    llm = LLM(
            model=path[model_type],
            trust_remote_code=True,
            swap_space=1,
            tensor_parallel_size=1 if "7b" in model_type or "8b" in model_type else torch.cuda.device_count(),
            max_model_len=8192,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(path[model_type])
    return llm, tokenizer


def _trunc_to_first_boxed(string):
        idx = string.find("\\boxed")
        if idx < 0:
            idx = string.find("\\fbox")
            if idx < 0:
                return string # if there is no box, finish cleaning

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1
        
        if left_brace_idx is None or right_brace_idx is None:
            return string

        return string[:right_brace_idx+2] # include } and $


def generate_sample_batch(llm, question_list, n, stop=["---", "\n\n\n", "\nQ: ", "\nA: ", "```"], max_tokens=1024):
        
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                        temperature=args.temperature,
                                        n=n,
                                        stop=stop,
                                        repetition_penalty=1.1)
        outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
        completions = [output.outputs[0].text.rstrip() for output in outputs]
        return completions


import signal
def evaluate_one_mcts_iteration(df):
    df = pd.DataFrame(df)

    from evaluation_utils.parallel_evaluation import parallel_eval_mcts
    df = parallel_eval_mcts(df, 20)
    # df = unparallel_eval(df)

    signal.signal(signal.SIGALRM, signal.SIG_DFL)

    return df

def check_condition(completion, task, step_num):
    if task == "math":
        no_step_mark = f"Step {step_num+1}:" not in completion
        no_box = "\\boxed" not in completion
        return no_box
    else:
        # first_step_end = completion.find("#")
        # first_step = completion[:first_step_end]
        # no_input_in_first_step = "input" not in first_step

        last_step_start = completion.rfind("#")
        last_step = completion[last_step_start:]
        no_print_in_last_step = "print" not in last_step
        # return no_step_mark or no_input_in_first_step or no_print_in_last_step
        no_comment = "#" not in completion
        # no_print = "print" not in completion
        return no_comment or no_print_in_last_step

def check_completion_format(completions, task, step_nums):
    completions = pd.DataFrame({"completion": completions})
    
    completions["format_error"] = completions.apply(lambda row, idx=0: check_condition(row["completion"], task, step_nums[idx]), axis=1)
    # print(completions.index)
    invalid_indices = completions[completions["format_error"]==True].index
    # print(invalid_indices)
    return invalid_indices


from copy import deepcopy
def generate_one_iteration(llm, tokenizer, df, task, setup, n_resp_per_inst, n, model_type):

    df = df.to_dict("records")

    icl = []
    icl_prompts = []
    from icl.code_icl import CODE_ICL_COT
    from icl.math_icl import MATH_ICL_COT

    template = {
            "math": "Solve the following math problem step by step with explicit 'Step x:' marks.\nSimplify your answer as much as possible. End your response with 'The final answer is: \\boxed{Your Answer}'.\n\nHere are some examples, please strictly follow the same format:\n\n[[incontext_examples]]\n---\nQ: [[question]]\n",
            # "code": "Solve the following coding problem step by step. Keep your explanations as clear and simple as possible. Divide essential code blocks into solution steps, marked explicitly with `# Step x:`. Ensure a balanced level of detail—not overly granular with one line per step, nor too broad with multiple functions per step. Typically, code blocks separated by \n\n indicate different steps, and a single function, loop, or if-else block could serve as a critical step. Begin with Step 1, which involves importing necessary libraries and receiving test inputs with `input()`. Conclude with the final step, which should solely display the results using `print()`, with all operations completed beforehand.\n\nHere are some examples:\n\n[[incontext_examples]]\n---\nQ: [[question]]\n" # 0.63125
            # "code": "Solve the following coding problem step by step. Keep your code as clear and simple as possible. Divide essential code blocks into solution steps, marked explicitly with `# Step x:`, each ensuring a balanced level of detail. Note, however, that the division of steps may also depend on the complexity of the program. For example, if the code is complex and consists of several functions, then each function should be a step as a whole; otherwise, if the code is rather simple, a step can be more fine-grained, such as a loop, an if-else branch, or even a critical one-line operation. But anyway, Step 1 always involves importing necessary libraries and receiving test inputs with `input()`, and the final step should always only contain `print()` to output the results, with all operations completed beforehand.\n\nHere are some examples:\n\n[[incontext_examples]]\n---\nQ: [[question]]\n"
            "code": "Solve the following coding problem. Keep your code as clear and simple as possible. Add comments using '#' before lines to improve the readability. Please write as more comments as possible. If a code block is very complex and contains many lines of code, you should decompose the code block by commenting its components, such as for-loop, if-else branch, or critical one-line operations.\n\nHere are some examples:\n\n[[incontext_examples]]\n---\nQ: [[question]]\n"
            # "code": "Solve the following coding problem step-by-step.\nSimplify your answer as much as possible. Present your code within a single code block. You need an input() function to receive inputs for testing. End your response by printing out the results, either using print rather than return within the class/function or add an extra print command.\n\nQ: [[question]]\n"
        }
    
    user_suffix = "Solve the math problem step-by-step with explicit 'Step x:' marks.\nSimplify your answer as much as possible. End your response with 'The final answer is: \\boxed{Your Answer}'." if task == "math" else "Add comments using '#' before lines to improve the readability. Please write as more comments as possible."
    # assistant_prefix = "A: Step 1:" if task != "code" else "A: ```python\n# Step 1:"
    assistant_prefix = "A: " if task != "code" else "A: ```python\n# Import necessary libraries and read inputs\n"

    for ex in {"code": CODE_ICL_COT, "math": MATH_ICL_COT}[task]:
        icl.append(f"Q: {ex['prompt']}\nA: {ex['completion']}")
        icl_prompts.append(ex['prompt'].strip().strip(".").strip().replace(" ", ""))
    incontext_examples = "\n---\n".join(icl)

    instructions = []
    index_steps_to_inst = []
    index_steps_to_comp = []
    step_nums = []
    connector = "\n" if task == "math" else ""
    for inst_idx, d in enumerate(df):

        tmp_icl = deepcopy(icl)

        if d["prompt"].strip().strip(".").strip().replace(" ", "") in icl_prompts: 
            idx = icl_prompts.index(d["prompt"].strip().strip(".").strip().replace(" ", ""))
            tmp_icl.pop(idx)
            assert len(tmp_icl) == len(icl_prompts)-1

        incontext_examples = "\n---\n".join(tmp_icl)
        for completion_idx, completion_steps in enumerate(d["steps"][:n_resp_per_inst]):
            if "rollouts" in df[inst_idx].keys(): # skip previous completions where we have sampled continuations
                if f"completion {completion_idx}" in df[inst_idx]["rollouts"].keys(): # we have already saved the mcts traj for this completion
                    continue

            for step_num in range(1, len(completion_steps)): # we do not sample for the last step
                partial_response = connector.join(completion_steps[:step_num])
                partial_response += f"\n\nStep {step_num+1}:" if task == "math" else "# "
                index_steps_to_inst.append(inst_idx)
                index_steps_to_comp.append(completion_idx)
                step_nums.append(step_num)

                instructions.append(tokenizer.apply_chat_template([
                        {"role":"user", "content":template[task].replace("[[incontext_examples]]", incontext_examples).replace("[[question]]", d["prompt"]) + user_suffix},
                        {"role":"assistant", "content":assistant_prefix + partial_response}
                    ], tokenize=False, add_generation_prompt=False))


    instructions = [instruction.rstrip("<|eot_id|>") for instruction in instructions]
    print(instructions[0])

    expanded_instructions = instructions * n
    expanded_step_nums = step_nums * n
    completions = [f"Step {expanded_step_nums[i]+1}: " + completion.strip() if task != "code" else "# " + completion.strip() 
                    for i, completion in enumerate(generate_sample_batch(llm, expanded_instructions, 1))]

    # re-roll until format correct
    num_iter = 10
    filtered_step_sums = deepcopy(expanded_step_nums)
    for i in range(num_iter):
        print(f"format check: iter {i+1}/{num_iter}")
        invalid_indices = check_completion_format(completions, task, filtered_step_sums)
        if len(invalid_indices) == 0:
            break
        filtered_step_sums = [expanded_step_nums[idx] for idx in invalid_indices]
        flattened_instructions = [inst for idx, inst in enumerate(expanded_instructions) if idx in invalid_indices]
        resampled_completions = [f"Step {filtered_step_sums[i]+1}: " + completion.strip() if task != "code" else "# " + completion.strip() 
                                    for i, completion in enumerate(generate_sample_batch(llm, flattened_instructions, 1))]
        # replace original completions
        for new_idx, ori_idx in enumerate(invalid_indices):
            completions[ori_idx] = resampled_completions[new_idx]
    print("\n"*3 + "-"*40)
    print(completions[0])

    if task == "math": # clean results
        completions = [_trunc_to_first_boxed(completion) for completion in completions]

    for i in range(len(instructions)):
        all_indices = [i + j*len(instructions) for j in range(n)]
        if "rollouts" not in df[index_steps_to_inst[i]].keys():
            df[index_steps_to_inst[i]]["rollouts"] = {}
        if f"completion {index_steps_to_comp[i]}" not in df[index_steps_to_inst[i]]["rollouts"].keys():
            df[index_steps_to_inst[i]]["rollouts"][f"completion {index_steps_to_comp[i]}"] = {}
        df[index_steps_to_inst[i]]["rollouts"][f"completion {index_steps_to_comp[i]}"][f"continuation {step_nums[i]}"] = [completions[idx] for idx in all_indices]
    
    df = pd.DataFrame(df)
    print(df.keys())

    if args.debug and task == "math":
        df = evaluate_one_mcts_iteration(df)

    if "task" in df.keys():
        del df["task"]
    
    return df
    



def read_data(file_path):
    try:
        df = pd.read_json(file_path, lines=True)
    except: # load huge datasets
        dataset = []
        import sys, json
        sys.set_int_max_str_digits(100000000) 
        # with open(file_path, "r", encoding="utf-8") as f:
        #     for line in f.readlines():
        #         dataset.append(json.loads(line))
        dataset = json.load(open(file_path, "r", encoding="utf-8"))
        df = pd.DataFrame(pd.DataFrame(dataset))
        # print(df)
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama-3.1-8b-inst")
    parser.add_argument("--task", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--data_per_gpu", type=int, default=500)
    parser.add_argument("--batch_id", type=int, default=0)
    parser.add_argument("--setup", type=str, default="onpolicy-inst", 
                        choices=["onpolicy-inst", "onpolicy-base", "single-strong", "iterative"])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=8) # math-shepherd # continuations
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_resp_per_inst", type=int, default=8)
    args = parser.parse_args()

    model_list = args.model_type.split("+")
    print(model_list)
    
    
    accelerator = Accelerator()

    
    file_list, origin_dataset = get_raw_data(args.bon_dataset)

    datasets = []
    # input_dir = "/home/test/test05/whb/data/o1_data"
    input_dir = f"/home/test/test05/ylf/prm/mcts_data/{args.setup}"
    output_dir_root = f"/home/test/test05/ylf/prm/mcts_data/{args.setup}" if not args.debug \
                        else f"/home/test/test05/ylf/prm/mcts_data_debug/{args.setup}"

    for task in [args.task]:
        
        start = args.batch_id*args.data_per_gpu
        end = (args.batch_id+1)*args.data_per_gpu
        end = min(end, 19409) if args.task == "code" else min(end, 32942)
        if not os.path.exists(os.path.join(input_dir, args.task, "merge", "all.json")):
            input_dir = f"/home/test/test05/ylf/prm/rollout_data/{args.setup}"
            dataset = read_data(os.path.join(input_dir, args.task, f"{start}-{end}.json")).to_dict(orient="records")
        else:
            dataset = read_data(os.path.join(input_dir, args.task, "merge", "all.json")).to_dict(orient="records")
            dataset = dataset[start:end]


        output_dir = os.path.join(output_dir_root, task)
        os.makedirs(output_dir, exist_ok=True)
        
        
        output_file_path = os.path.join(output_dir, f"{start}-{end}.json")

        df = pd.DataFrame(dataset) if not args.debug else pd.DataFrame(dataset[:1])
        df["task"] = [task]*len(df)

        if args.setup != "iterative":
            if os.path.exists(output_file_path) and not args.debug: # already finished
                continue
            llm, tokenizer = load_model(model_list[0])
            df = generate_one_iteration(llm, tokenizer, df, task, args.setup, args.n_resp_per_inst, args.n, args.model_type)
            df.to_json(output_file_path, orient="records", indent=4)
        else:
            model_idx_start = 0
            if os.path.exists(output_file_path):
                # hard code for iterative
                df = pd.read_json(output_file_path)
                print(df)
                print(df.iloc[0]["model"])
                model_idx_start = len(set(df.iloc[0]["model"]))
                assert model_idx_start > 0
                # we need 4 models in total, but save the df after each model finishing generation
                # check if it's this case; otherwise, exit, since all have been generated.
                print(f"4 models needed, {model_idx_start} models finished")

            for model_idx in range(model_idx_start, len(model_list)):
                if model_idx_start == 0 and model_idx > 0: # reload model
                    del llm.llm_engine
                    del llm, tokenizer
                    cleanup()
                llm, tokenizer = load_model(model_list[model_idx])
                df["task"] = [task]*len(df)
                df = generate_one_iteration(llm, tokenizer, df, task, args.setup, int(args.n/4), model_list[model_idx])
                df.to_json(output_file_path, orient="records", indent=4)
                print("saved", model_list[model_idx])
    
    print("data finished:", start, end)
    if args.debug and args.task == "math":
        print("avg_acc:", df["acc"].mean())
        print("var_acc:", df["acc"].var())