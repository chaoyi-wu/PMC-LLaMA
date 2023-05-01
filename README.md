# PMC-LLaMA
The official codes for "PMC-LLaMA: Continue Training LLaMA on Medical Papers"

[**Huggingface**](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B) 

[**Arxiv Version**]()

## Introduction:
We continue pre-training LLaMA on 4.8M PubmedCentral papers.

## Environment:
Simply set up the required environment as following:
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers,sentencepiece,datasets
```

## Quick Start:
Check `simple_test.py` for quickly use PMC-LLaMA or you can follow this folowing simple sample.

```
import transformers
import torch
tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
sentence = 'Hello, doctor' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=200, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))
```

## Data:
The raw training data can be dowloaded from [S2ORC](https://github.com/allenai/s2orc), filter out the papers with PubmedCentral IDs, and you can get the training data we use. 

We will also release a version of training data soon.

## Pre-training:
Check `training.py` and `training.sh` for re-produce our work. 

More details about how to fine-tune LLaMA can refer to [Finetune_LLAMA](https://github.com/chaoyi-wu/Finetune_LLAMA)

## Results:
| Method              | Setting             | USMLE(OOD/ID) | MedMCQA(ID) | PubMedQA(ID) |
|---------------------|---------------------|------------------|--------------|------------------|
| Human (pass)        | Manual*             | 50.0            | --            | 60.0           |
| Human (expert)      | Manual*             | 87.0            | 90.0         | 78.0           |
| InstructGPT-175B    | Zero-shot*          | 46.0            | 44.0         | 73.2           |
| ChatGPT             | Zero-shot*          | 57.0            | 44.7         | 63.9           |
| LLaMA-7B            | Zero-shot*          | 27.1            | 24.3         | 5.2             |
| LLaMA-33B           | Zero-shot*          | 43.4            | 30.3         | 1.8             |
| LLaMA-7B-Full  | Full fine-tuning   | 44.55/35.66     | 48.15        | 73.41          |
| PMC-LLaMA-7B-Full | Full fine-tuning | 44.70/40.61     | 50.54        | 69.53          |
| LLaMA-7B-PEFT  | PEFT               | 29.38/27.34     | 32.37        | 65.81          |
| PMC-LLaMA-7B$-PEFT | PEFT             | 30.64/28.52     | 34.33        | 68.23          |
Note that, the manual and zero-shot results with * are referred from [LMFLow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow).

## Downstream Training Curve:
<img src="https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/figures/training_curve.png"/>

## Zero-shot Cases:
<img src="https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/figures/zero-shot_cases.png"/>

## Acknowledge
Minimal LLaMA -- https://github.com/zphang/minimal-llama

alpaca -- https://github.com/tatsu-lab/stanford_alpaca

LMFLow -- https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

## Contact
If you have any question, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.

