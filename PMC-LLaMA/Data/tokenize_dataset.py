import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm

import datasets
import transformers
import json
import jsonlines
import csv
def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def sentence_make(sentence,reflect_array,special_tokens_list):
    new_sss = ''
    for i in range(len(sentence)):
        if reflect_array[i] == 'T':
            new_sss = new_sss + sentence[i]
        if reflect_array[i] == 'A':
            new_sss = new_sss + special_tokens_list[0] + sentence[i]
        if reflect_array[i] == 'B':
            new_sss = new_sss + special_tokens_list[1] + sentence[i]
    return new_sss
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,default='/Path/to/LLAMA_Model/tokenizer')
    parser.add_argument("--jsonl_path", type=str,default='/Path/to/PMC_filter.jsonl')
    parser.add_argument("--save_path", type=str,default='./Data_sample/PMC_OA_papers/preprocessor')
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--start", type=int, default = 0)
    parser.add_argument("--end", type=int, default = 1000000)
    args = parser.parse_args()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)
    special_tokens_list= ['[author]','[bib]']
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    elems = read_jsonl(args.jsonl_path)
    #ending_names = ["ending0", "ending1", "ending2", "ending3"]
    # i = 0
    total = 0
    i = 0
    with open("/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/Data_sample/PMC_OA_papers/Tokenized/name_list.csv","a",encoding="UTF-8",newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['PMCid'])
        for elem in tqdm.tqdm(elems):
            i = i+1
            if i<=args.start:
                continue
            sentence = elem['content']['text']
            if sentence is None:
                continue
            reflect_array = ['T' for _ in range(len(sentence))]
            if sentence == None:
                continue
            try:
                author_info = json.loads(elem['content']['annotations']['author'])
                for author in author_info:
                    start = int(author["start"])
                    end = int(author["end"])
                    reflect_array[start] = 'A'  
                    reflect_array[end] = 'A'
            except:
                pass
            try:
                bib_info = json.loads(elem['content']['annotations']['bibentry'])
                for bib in bib_info:
                    start = int(bib["start"])
                    end = int(bib["end"])
                    reflect_array[start] = 'B'  
                    reflect_array[end] = 'B'
            except:
                pass
            name = 'PMC' + elem['externalids']['pubmedcentral'] + '.npy'
            sentence = sentence_make(sentence,reflect_array,special_tokens_list)
            save_dir = '/nvme/zhangruipeng/wuchaoyi/Finetune_llama_by_wucc/Data_sample/PMC_OA_papers/Tokenized/' + name 
            writer.writerow([name])
            case = np.array(tokenizer.encode(sentence))
            np.save(save_dir, case)
            if i>=args.end:
                break
            #print(total)

if __name__ == "__main__":
    main()


#python tokenize_dataset.py --start 0 --end 500000
#python tokenize_dataset.py --start 500000 --end 1000000
#python tokenize_dataset.py --start 1000000 --end 1500000
#python tokenize_dataset.py --start 1500000 --end 2000000
#python tokenize_dataset.py --start 2000000 --end 2500000
#python tokenize_dataset.py --start 2500000 --end 3000000
#python tokenize_dataset.py --start 3000000 --end 3500000
#python tokenize_dataset.py --start 3500000 --end 4000000
#python tokenize_dataset.py --start 4000000 --end 4500000
#python tokenize_dataset.py --start 4500000 --end 5000000