import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import json
import PIL
import numpy as np
import torch.nn.functional as F
import tqdm
import transformers
import os
import copy
import random
import jsonlines
import pandas as pd
import csv
# def read_jsonl(path):
#     # Manually open because .splitlines is different from iterating over lines
#     with open(path, "r") as f:
#         for line in f:
#             yield json.loads(line)

def read_jsonl(path,eval_num):
    df = pd.read_csv(path)
    elems = list(df['PMCid'])
    random.shuffle(elems)
    Eval_elems = elems[:eval_num]
    Train_elems = elems[eval_num:]
    with open('Eval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for sublist in Eval_elems:
            writer.writerow([sublist])
    with open('Train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for sublist in Train_elems:
            writer.writerow([sublist])
    return Train_elems,Eval_elems  
      
class PaperDataset(Dataset):
    def __init__(self, root_path, dir_name, seq_length = 512, voc_size = 32000,keep_author = True, keep_bib = True):
        df = pd.read_csv(root_path+dir_name)
        elems = list(df['PMCid'])
        self.root_path = root_path
        self.elems = elems
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.keep_author = keep_author
        self.keep_bib = keep_bib
    
    def __len__(self):
        return len(self.elems)
    
    def __getitem__(self,idx):
        npy_path =  self.root_path + '/'+ self.elems[idx]
        data = np.load(npy_path)
        input_id = self.random_subsection(data)
        label = copy.deepcopy(input_id)
        return dict(input_ids=input_id, labels=label)
    
    def random_subsection(self, arr):
        if self.keep_author:
            arr = arr[arr!=self.voc_size]
        if self.keep_bib:
            arr = arr[arr!=(self.voc_size+1)]
        if len(arr) < self.seq_length:
            arr = np.pad(arr, (0, self.seq_length - len(arr)), 'constant', constant_values=2)
        if len(arr) - self.seq_length == 0:
            start = 0
            return arr[start:start+self.seq_length]
        start = np.random.randint(0, len(arr) - self.seq_length)
        while(np.sum(arr[start:start+self.seq_length] == self.voc_size) %2 !=0):
            start = np.random.randint(0, len(arr) - self.seq_length)
        #arr = torch.tensor(arr[start:start+self.seq_length])
        return arr[start:start+self.seq_length]
    
#Train_elems, Eval_elems = read_jsonl('/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/Data_sample/PMC_OA_papers/Tokenized/name_list.csv',10000)
#dataset = PaperDataset('/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/Data_sample/PMC_OA_papers/Tokenized', Train_elems)
#print(dataset[0])
