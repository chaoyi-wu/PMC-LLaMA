'''
CUDA_VISIBLE_DEVICES=4,5,6,7 python medqa_inference.py \
    --model-name-or-path path/to/pmc_llama_model \
    --data-path /path/to/test.jsonl \
    --write-dir /path/to/inferenced_result_dir
'''

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers

from typing import Dict, Optional, Sequence
import argparse
import jsonlines
from tqdm import tqdm
from functools import partial
import os


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--write-dir', type=str)
    args = parser.parse_args()
    return args


def construct_spedical_tokens_dict() -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
# enddef



def inference_on_one(input_str: Sequence[str], model, tokenizer) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=1000,
        top_k=50
    )

    # topk_output = model.generate(
    #     **model_inputs,
    #     max_new_tokens=1000,
    #     top_k=50
    # )

    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]



def read_jsonl(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def prepare_data(data_list: Sequence[dict], model, tokenizer) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        sample_id = data_entry['sample_id']

        data_list[_idx]['pmc_input'] = prompt_input.format_map(data_entry) if data_entry.get("input", "") != "" else prompt_no_input.format_map(data_entry)
    # endfor
    return data_list
# enddef


if __name__ == '__main__':
    print('h')

    args = parse_args()

    print(f"\033[32mPrepare Data\033[0m")
    data_list = read_jsonl(args.data_path)
    fn = partial(prepare_data, model=None, tokenizer=None)
    inference_data = fn(data_list)

    print(f"\033[32mLoad Checkpoint\033[0m")
    model = transformers.LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        #cache_dir=training_args.cache_dir,
        model_max_length=400,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = construct_spedical_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.cuda()

    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        sample_id = data_entry['sample_id']
        input_str = [
            data_entry['pmc_input']
        ]
        output_str = inference_on_one(input_str, model, tokenizer)
        with open(os.path.join(args.write_dir, f"{sample_id}.txt"), 'w') as f:
            f.write(output_str)
    # endfor

