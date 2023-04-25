import argparse
import os
import json
import math
import tqdm.auto as tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import datasets
import transformers
from Dataset.Paper_dataset import PaperDataset,read_jsonl
from transformers import Trainer
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
import os
    
def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x))
    
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLAMA_Model/llama-13b-hf")
    is_lora: Optional[bool] = field(default=False)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    Dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # Train_dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # Eval_dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    #Train_elems, Eval_elems = read_jsonl(data_args.Dataset_path + '/name_list.csv',eval_num = 10000)
    Train_dataset = PaperDataset(data_args.Dataset_path, '/Train.csv')
    Eval_dataset = PaperDataset(data_args.Dataset_path, '/Eval.csv')
    
    
    print("Setup Model")
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
    )
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args=training_args,
                      )
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
      


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_pp_peft_trainer.py
