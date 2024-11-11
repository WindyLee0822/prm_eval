from datasets import load_from_disk, Dataset
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
from functools import partial
from copy import deepcopy
import re
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import deepspeed
import os
from accelerate.utils import gather_object

def get_raw_data(dataset_name, process_inst=False):
    if dataset_name == 'gsm8k':
        file_list = [
            '/home/test/test05/lwd/mcts-data/testset/gsm8k-plus-llama3.1-8b-inst-128.json',
            '/home/test/test05/lwd/mcts-data/testset/gsm8k-plus-llama3-70b-inst-128.json',
            '/home/test/test05/lwd/mcts-data/testset/gsm8k-plus-Eurux-8x22b-nca-128.json',
        ]
        origin_dataset = load_from_disk('/home/test/test05/lwd/hf-dataset-download/GSM-Plus')['testmini']
        origin_dataset = [d for d in origin_dataset][:500]

    elif dataset_name == 'qa':
        file_list = [
            '/home/test/test05/lwd/mcts-data/testset/qa-llama3.1-8b-inst-128.json',
            '/home/test/test05/lwd/mcts-data/testset/qa-llama3-70b-inst-128.json',
            '/home/test/test05/lwd/mcts-data/testset/qa-Eurux-8x22b-nca-128.json',
        ]
        origin_dataset = json.load(open('/home/test/test05/lwd/mcts-data/testset/theorem_qa.json'))
        origin_dataset = [d for d in origin_dataset if d['Picture'] == None]

    elif dataset_name == 'math':
        file_list = [
            '/home/test/test05/lwd/mcts-data/testset-0.5/math-Eurux-8x22b-nca-64.json',
            '/home/test/test05/lwd/mcts-data/testset-0.5/math-Meta-Llama-3-70B-Instruct-64.json',
            '/home/test/test05/lwd/mcts-data/testset-0.5/math-llama3.1-8b-inst-64.json',

        ] if process_inst else [
            # '/home/test/test05/lwd/mcts-data/testset/math-o1-sft-64.json'
            '/home/test/test05/ylf/prm_eval/testset/math-Eurux-8x22b-nca-64.json',
            '/home/test/test05/ylf/prm_eval/testset/math-Meta-Llama-3-70B-Instruct-64.json',
            '/home/test/test05/ylf/prm_eval/testset/math-llama3.1-8b-inst-64.json',
        ]
        path = '/home/test/test05/lwd/mcts-data/testset/MATH500.jsonl'
        with open(path) as f:
            origin_dataset = [json.loads(line) for line in f]

    return file_list, origin_dataset

def get_tokenizer(tokenizer_path, ref_tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.apply_chat_template([{'role': 'user', 'content': ' '}], add_generation_prompt=True, tokenize=False)
    except:
        print('WARNING:your tokenizer does not have a default template')

    if ref_tokenizer_path!=None:
        ref_tokenizer = AutoTokenizer.from_pretrained(ref_tokenizer_path)
        if not ref_tokenizer.pad_token:
            ref_tokenizer.pad_token = ref_tokenizer.eos_token
        try:
            ref_tokenizer.apply_chat_template([{'role': 'user', 'content': ' '}], add_generation_prompt=True,
                                          tokenize=False)
        except:
            print('WARNING:your ref_tokenizer does not have a default template')
    else:
        ref_tokenizer=None
    return tokenizer, ref_tokenizer


def set_special_token_ids(prm_token, good_token, bad_token, tokenizer):

    prm_token_id = tokenizer.encode(f"{prm_token}")[-1] if prm_token else None
    good_token_id = tokenizer.encode(f"{good_token}",add_special_tokens=False)[-1]
    bad_token_id = tokenizer.encode(f"{bad_token}",add_special_tokens=False)[-1]

    return prm_token_id, good_token_id, bad_token_id


from transformers import PreTrainedModel, AutoConfig, AutoModel
class LlamaRewardModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.value_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.value_head(hidden_states).squeeze(-1)

        return rewards



def init_ds_models(type, load, ref_load, bon_dataset):
    if 'dpo' in type:
        model = AutoModelForCausalLM.from_pretrained(load).cuda()  # torch_dtype=torch.bfloat16)
        ds_engine = deepspeed.init_inference(model,
                                             tensor_parallel={"tp_size": 1},
                                             dtype=torch.bfloat16)
        model = ds_engine.module
        model.eval().requires_grad_(False)
    elif type=='prm-value':
        model = LlamaRewardModel.from_pretrained(load).cuda()  # torch_dtype=torch.bfloat16)
        ds_engine = deepspeed.init_inference(model,
                                             tensor_parallel={"tp_size": 1},
                                             dtype=torch.bfloat16)
        model = ds_engine.module
        model.eval().requires_grad_(False)
    elif type=='prm-llm':
        model = AutoModelForCausalLM.from_pretrained(load).cuda()  # torch_dtype=torch.bfloat16)
        ds_engine = deepspeed.init_inference(model,
                                             tensor_parallel={"tp_size": 1},
                                             dtype=torch.bfloat16)
        model = ds_engine.module
        model.eval().requires_grad_(False)

    # load ref only when we haven't saved its logits; otherwise we can directly load forwarded logits and no need for another pass
    dir_name = ref_load.split('/')[-1] if ref_load else "placeholder"
    ref_logits_dir = f"/home/test/test05/ylf/prm_eval/ref_logits/{dir_name}"
    os.makedirs(os.path.join(ref_logits_dir, bon_dataset), exist_ok=True)
    ref_logits_path_list = [f"{ref_logits_dir}/{bon_dataset}/{testset_generator}.json" 
                            for testset_generator in ["eurux-8x22b-nca","llama3.1-70b-inst","llama3.1-8b-inst"]] #'math-o1-sft-64']]#
    if 'dpo' in type and ref_load != None and any([not os.path.exists(ref_logits_path) for ref_logits_path in ref_logits_path_list]):
        ref_model = AutoModelForCausalLM.from_pretrained(ref_load).cuda()  # torch_dtype=torch.bfloat16)
        ref_ds_engine = deepspeed.init_inference(ref_model,
                                                    tensor_parallel={"tp_size": 1},
                                                    dtype=torch.bfloat16)
        ref_model = ref_ds_engine.module
        ref_model.eval().requires_grad_(False)
    else:
        ref_model = None

    return model, ref_model, ref_logits_path_list



def load_data(file_name, origin_dataset):
        
    queries = []
    if file_name.endswith('json'):
        cur_data = json.load(open(file_name))
    else:
        with open(file_name) as f:
            cur_data = [json.loads(line) for line in f]

    if 'gsm' in file_name:
        cur_data = cur_data[:500]

    cur_queries = deepcopy(cur_data)
    # print(cur_queries[:10])
    if "reference" in cur_queries[0].keys():
        return cur_queries

    assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(cur_queries))
    for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
        if 'problem' in ori:
            assert 'question' in data and data['question'] == ori['problem'] or 'problem' in data and data[
                'problem'] == ori['problem']
        elif 'question' in ori:
            assert 'question' in data and data['question'] == ori['question'] or 'problem' in data and data[
                'problem'] == ori['question']
        else:
            assert 'question' in data and data['question'] == ori['Question'] or 'problem' in data and data[
                'problem'] == ori['Question'] or 'Question' in data and data['Question'] == ori['Question'], (
            data.keys(), ori.keys())
        assert len(data['responses']) == 128 or len(data['responses']) == 64, (
            file_name, len(data['responses']))
        if len(data['responses']) > 64:
            data['responses'] = data['responses'][:64]
        for response_dict in data['responses']:
            queries.append({
                'idx': idx,
                'prompt': data['question'] if 'question' in data else (
                    data['problem'] if 'problem' in data else data['Question']),
                'response': response_dict['text'],
                'solution': ori['solution'] if 'solution' in ori else str(ori['Answer']),
            })
    return queries



def dpo_data_collator(example, tokenizer, special_ids, accelerator, type):

    inputs = []
    idx,reward_idx = [],[]
    special_tokens = []
    label_mask = []
    for d in example:
        input_ids = tokenizer.apply_chat_template([
            {"role":"user", "content":d["query"]},
            {"role":"assistant", "content":"\n\n".join(d["answer"])},
        ], tokenize=True, add_generation_prompt=False)
        inputs.append(torch.tensor(input_ids))

        # TODO: the readability is low; should directly locate steps by its length
        cur_special_ids = []
        if "orm" not in type:
            for idd,id in enumerate(input_ids[1:]):
                if id in special_ids[:-1]:
                    flag = 0
                    for sub_i in range(idd + 1, min(len(input_ids)-1, idd + 6)):
                        if input_ids[1:][sub_i] == special_ids[-1]:
                            flag = 1
                            break
                        if input_ids[1:][sub_i] in special_ids[:-1]:
                            break
                    if flag and idd-1>=0:
                        cur_special_ids.append(idd-1)
        else:
            cur_special_ids.append(0)
        cur_special_ids.append(len(input_ids[1:])-1)

        # assert len(cur_special_ids)==len(d['labels'])
        special_tokens.append(torch.tensor(cur_special_ids))
        
        label_mask.append(torch.tensor( [0]*cur_special_ids[0]+[1]*(len(input_ids)-cur_special_ids[0]) ))
        idx.append(d['idx'])
        reward_idx.append(d['reward_idx'])

    labels = pad_sequence(inputs, padding_value=-100, batch_first=True)
    inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask = (inputs!=tokenizer.pad_token_id)
    label_mask = pad_sequence(label_mask, padding_value=0, batch_first=True)
    special_tokens = pad_sequence(special_tokens, padding_value=-100, batch_first=True)

    return {
        'input_ids': inputs.int().to(accelerator.device),
        'attention_mask': attention_mask.int().to(accelerator.device),
        'labels':labels.int().to(accelerator.device),
        'label_mask':label_mask.to(accelerator.device),
        'special_tokens':special_tokens.to(accelerator.device),
        'idx':torch.tensor(idx).to(accelerator.device),
        'reward_idx':torch.tensor(reward_idx).to(accelerator.device)
    }

def prm_data_collator(example, tokenizer, special_ids, accelerator):
    from itertools import chain
    inputs = []
    idx,reward_idx = [],[]
    special_tokens = []
    label_mask = []
    attention_mask = []
    prm_token_id = special_ids["prm_token_id"]
    for d in example:

        query_str = tokenizer.apply_chat_template([{"role": "user", "content": d["query"]}], tokenize=False, add_generation_prompt=True)
        query_ids = tokenizer(query_str, padding=False, add_special_tokens=False).input_ids
        split_tokens = []

        answer_ids = tokenizer(d["answer"], add_special_tokens=False).input_ids
        input_ids = query_ids + split_tokens + list(chain(*[lst + [prm_token_id] for lst in answer_ids])) + [prm_token_id, tokenizer.eos_token_id]
        inputs.append(torch.tensor(input_ids))
        
        attn_mask = [1]*len(input_ids)
        for idd, inp_id in enumerate(input_ids):
            if inp_id in special_ids.values():
                attn_mask[idd] = 0
        attention_mask.append(torch.tensor(attn_mask))

        cur_special_ids = []

        for idd,id in enumerate(input_ids):
            if id == prm_token_id:
                cur_special_ids.append(idd)
        # remove the first prm_token_id between query and answr
        # if not args.use_osv_template:
        #     cur_special_ids = cur_special_ids[1:]

        if len(cur_special_ids)<1:
            print(tokenizer.tokenize(template.format(query=d['query'],answer=d['answer']),
                                        add_special_tokens=False),input_ids,prm_token_id)
            raise ValueError

        special_tokens.append(torch.tensor(cur_special_ids))
        idx.append(d['idx'])
        reward_idx.append(d['reward_idx'])

    inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask = pad_sequence(attention_mask, padding_value=tokenizer.pad_token_id, batch_first=True)

    special_tokens = pad_sequence(special_tokens, padding_value=-100, batch_first=True)

    return {
        'input_ids': inputs.int().to(accelerator.device),
        'attention_mask': attention_mask.int().to(accelerator.device),
        'special_tokens':special_tokens.long().to(accelerator.device),
        'idx':torch.tensor(idx).to(accelerator.device),
        'reward_idx':torch.tensor(reward_idx).to(accelerator.device)
    }




def get_dataloader(type, queries, batch_size, tokenizer, ref_tokenizer, step_mark_ids, prm_special_ids, accelerator):

    for idx, data in enumerate(queries):
        data['reward_idx'] = idx
        data["query"] = data["prompt"]
        steps = re.split('Step \d+:', data['response'])
        steps = [step.strip().replace('Step', 'step') for step in steps if step.strip() != '']
        steps = [f'Step {id + 1}: ' + step for id, step in enumerate(steps) if step.strip() != '']
        data["answer"] = steps
        data['logprobs'] = 0

    if accelerator.is_local_main_process:
        print('Dataset Example:')
        print(queries[0])

    dataset = Dataset.from_pandas(pd.DataFrame.from_records(queries))

    data_collator = {'dpo': dpo_data_collator, 'dpo-orm': dpo_data_collator, 'prm-value': prm_data_collator, 'prm-llm': prm_data_collator}
    special_ids = {'dpo': step_mark_ids, 'dpo-orm': step_mark_ids, 'prm-value': prm_special_ids, 'prm-llm': prm_special_ids}

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                            collate_fn=partial(data_collator[type], tokenizer=tokenizer, special_ids=special_ids[type], accelerator=accelerator, type=type))
    assert ref_tokenizer!=None
    ref_dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                collate_fn=partial(dpo_data_collator, tokenizer=ref_tokenizer, special_ids=special_ids[type], accelerator=accelerator, type=type)) if ref_tokenizer!=None else None

    return dataloader, ref_dataloader


def devide_dataloader_to_devices(dataloader, accelerator, local_rank):
    tmp = []
    dataloader_per_device = [[] for _ in range(accelerator.num_processes)]
    for iteration, data in enumerate(dataloader):
        tmp.append(data)
        steps = iteration + 1
        if steps % accelerator.num_processes == 0:
            for i in range(accelerator.num_processes):
                dataloader_per_device[i].append(tmp[i])
            tmp = []

    assert len(tmp)==0
    dataloader = dataloader_per_device[local_rank]
    return dataloader


def get_logps(model,inputs):
    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
    labels = inputs['labels'][:, 1:].clone().long()
    logits = logits[:, :-1, :]
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    loss_mask = labels != -100
    per_token_logps = per_token_logps * loss_mask

    return per_token_logps



def compute_beta(rewards, method):

    ori_shape = rewards.shape
    if method == "constant":
        beta = 1
    
    if method == "weighted":
        beta = 1 / torch.ones_like(rewards).cumsum(-1).cuda()
        assert beta.shape == rewards.shape, (beta.shape, rewards)
    elif method == "weighted exponential decay":
        beta = 0.95 ** (torch.ones_like(rewards).cumsum(-1) - 1).cuda()
        assert beta.shape == rewards.shape, (beta.shape, rewards)
    elif method == "length normalized":
        beta = 1 / (rewards.cumsum(-1).argmax(-1) + 1).unsqueeze(-1).cuda()
        assert beta.shape[0] == rewards.shape[0] and beta.shape[1] == 1, (beta.shape, rewards.shape)

    rewards = beta * rewards
    
    assert rewards.shape == ori_shape
    
    return rewards



def get_reward(model, inputs, type, accelerator, ref_per_token_logps=None, good_token_id=None, bad_token_id=None):
    if 'dpo' in type:
        with torch.no_grad():
            per_token_logps = get_logps(model,inputs)

        cur_index = torch.where(inputs['special_tokens']==-100, 0, inputs['special_tokens']) #-1 因为label向前移了一位

        all_rewards = {}
        # for ref_setup in ["w/ ref", "w/o ref"]:
        for ref_setup in ["w/ ref"]:
            all_rewards[ref_setup] = {}
            raw_reward = per_token_logps - ref_per_token_logps if ref_setup == "w/ ref" else per_token_logps
            raw_reward = raw_reward * inputs['label_mask'][:, 1:]

            # for beta_method_outer in ["constant", "weighted", "weighted exponential decay", "length normalized"]:
            for beta_method_outer in ['constant',"length normalized"] if type != "dpo-orm" else ['constant']:
                
                beta_reward_before_scaling = compute_beta(raw_reward, beta_method_outer)
                for coef in [0.001, 0.005, 0.01, 0.05]:
                    beta_method = f"{beta_method_outer}+coef-{coef}"
                    all_rewards[ref_setup][beta_method] = {}

                    beta_reward = coef * beta_reward_before_scaling

                    beta_reward = beta_reward.cumsum(-1)

                    beta_reward = beta_reward.gather(dim=-1, index=cur_index[:, 1:])
                    beta_reward = torch.where(inputs['special_tokens'][:, 1:] == -100, 1e3, beta_reward)

                    for reward_approach in ["single", "accumulative"] if type != "dpo-orm" else ["orm"]:
                        if reward_approach == "single":
                            rewards = beta_reward - torch.cat([torch.zeros(beta_reward.shape[0],1, device=beta_reward.device),
                                                beta_reward[:, :-1]], dim=-1)
                            rewards = torch.where(inputs['special_tokens'][:, 1:] == -100, 1e3, rewards)
                        else:
                            rewards = beta_reward.clone()
                        # rewards = gather_object(rewards.tolist())
                        rewards = gather_object(rewards)
                        all_rewards[ref_setup][beta_method][reward_approach] = rewards
                        # all_rewards[ref_setup][reward_approach] = rewards

    elif 'prm' in type:
        with torch.no_grad():
            if type=='prm-value':
                rewards  = model(input_ids = inputs['input_ids'],attention_mask = inputs['attention_mask'])
            elif type=='prm-llm':
                logits = model(input_ids = inputs['input_ids'],attention_mask = inputs['attention_mask']).logits[:, :, [good_token_id,bad_token_id]]
                rewards = logits.softmax(dim=-1)[:, :, 0]

        cur_index = torch.where(inputs['special_tokens'] == -100, 0,inputs['special_tokens'])
        rewards = rewards.gather(dim=-1, index=cur_index)
        rewards = torch.where(inputs['special_tokens']==-100, 1e3, rewards)
        # rewards = gather_object(rewards.tolist())
        rewards = gather_object(rewards)
        all_rewards = {"w/o ref": {"constant": {"single": rewards}}}
        
    
    reward_idxes = accelerator.gather(inputs['reward_idx'])
    return all_rewards, reward_idxes


def manipulate_rewards(all_rewards, queries, reward_idxes, accelerator):

    for ref_setup in all_rewards.keys():
        for beta_method in all_rewards[ref_setup].keys():
            for reward_approach in all_rewards[ref_setup][beta_method].keys():
                ori_rewards = all_rewards[ref_setup][beta_method][reward_approach]

                different_calculations = {}
                # for method in ["min", "sum", "mean"]:
                for method in ["min", "sum"]:

                    if method == "min":
                        rewards = [reward.min(-1).values for reward in ori_rewards]
                    elif method == "sum":
                        rewards = [torch.where(reward==1e3,0,reward).sum(-1)  for reward in ori_rewards]
                    elif method == "mean":
                        rewards = [torch.where(reward==1e3,0,reward).sum(-1)/torch.where(reward==1e3,0,1).sum(-1) for reward in ori_rewards]

                    # rewards = gather_object(rewards)
                    different_calculations[method] = rewards

                    assert len(rewards) == len(reward_idxes)
                    for reward, reward_idx in zip(rewards, reward_idxes):
                        if "reward" not in queries[int(reward_idx)].keys():
                            queries[int(reward_idx)]["reward"] = {}
                        if ref_setup not in queries[int(reward_idx)]["reward"].keys():
                            queries[int(reward_idx)]["reward"][ref_setup] = {}
                        if beta_method not in queries[int(reward_idx)]["reward"][ref_setup].keys():
                            queries[int(reward_idx)]["reward"][ref_setup][beta_method] = {}
                        if reward_approach not in queries[int(reward_idx)]["reward"][ref_setup][beta_method].keys():
                            queries[int(reward_idx)]["reward"][ref_setup][beta_method][reward_approach] = {}
                        # queries[int(reward_idx)]["reward"]["all_steps"] = [rew.tolist() for rew in ori_rewards]
                        queries[int(reward_idx)]["reward"][ref_setup][beta_method][reward_approach][method] = reward.item()

                all_rewards[ref_setup][beta_method][reward_approach] = different_calculations
        
    return all_rewards, queries


def split_query(completions, n, N=16): # extract top-n logprob completion for each query
    splitted_completions = []
    for idx in range(int(len(completions) / N)):
        samples = [sample for sample in completions if sample["idx"] == idx]
        samples = sorted(samples, key=lambda x: x["logprobs"], reverse=True)
        splitted_completions.append(samples[:n])
    return splitted_completions

def best_of_n(splitted_completions,key='reward'):
    selected_completions = []
    for n_completions_per_query in splitted_completions:
        n_completions_per_query = sorted(n_completions_per_query, key=lambda x: x[key], reverse=True)
        assert all([n_completions_per_query[0][key] >= completion[key] for completion in n_completions_per_query])
        selected_completions.append(n_completions_per_query[0])
    return selected_completions