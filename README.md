# PMC-LLaMA

The official codes for "PMC-LLaMA: Towards Building Open-source Language Models for Medicine". 

<!-- vim-markdown-toc GFM -->

* [Latest News](#latest-news)
* [Environment](#environment)
* [Quick Start](#quick-start)
* [Training](#training)
* [Results](#results)
    * [QA Benchmark](#qa-benchmark)
    * [Zero-shot Cases](#zero-shot-cases)
* [Acknowledge](#acknowledge)
* [Contact](#contact)

<!-- vim-markdown-toc -->

[**Arxiv Version**](https://arxiv.org/abs/2304.14454)

We prove that medical LLM should be first pretrained with domain corpus, and then tuned with instructions following dataset.

We have released The latest model **PMC_LLaMA_13B** finetuned on our instructions the following dataset.
It has shown a better ability to follow user instructions than MedLLaMA_13B.

<img src=./figures/teaser.png width="50%">

Similarly, it can be easily loaded with:

```python
import transformers
import torch
tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')
```
Hereby we present PMC_LLaMA's versions and briefs.

[MedLLaMA_13B](https://huggingface.co/chaoyi-wu/MedLLaMA_13B) is pretrained on medical corpus, and [PMC_LLaMA_13B](https://huggingface.co/axiong/PMC_LLaMA_13B) is further finetuned based on that.

| Version | Link | Brief | Release Date |
| --- | --- | --- | --- |
|MMed-Llama-3 ![](./figures/new.gif) | https://huggingface.co/Henrychur/MMed-Llama-3-8B | Latest Pretrained Multilingual LLM on Llama-3 | 2024/05/22 |
| MMedLM  | https://github.com/MAGIC-AI4Med/MMedLM | Further Pretrained Multilingual LLM | 2024/02/21 |
| PMC_LLaMA_13B | https://huggingface.co/axiong/PMC_LLaMA_13B | Instruction Tuned | 2023/09/01 |
| MedLLaMA_13B | https://huggingface.co/chaoyi-wu/MedLLaMA_13B | Pre-training LLaMA on 4.8M PubmedCentral papers and Medical Books | 2023/05/01 |
| PMC_LLaMA_7B_10_epoch | https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B_10_epoch | Similar to PMC_LLaMA_7B but trained 10 epochs | 2023/05/01 |
| PMC_LLaMA_7B | https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B | LLaMA-7b finetuned with PMC papers for 5 epochs | 2023/04/25 |


## Latest News
We have released a new report genration metrics [RaTEScore](https://arxiv.org/abs/2406.16845). We strongly believe to promote the develop a generative-based medical foundation models, developing a robust and reliable metric is a critical and foundation step. 

## Environment
Simply set up the required environment as following:
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers=4.28.1, sentencepiece, datasets
```

## Quick Start
Check `simple_test.py` for quickly use PMC-LLaMA or you can follow this folowing simple sample.

```python
import transformers
import torch
tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')
model.cuda()  # move the model to GPU

prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)

example = {
    "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    "input": (
        "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
        "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
        "She otherwise feels well and is followed by a doctor for her pregnancy. "
        "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
        "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
        "Which of the following is the best treatment for this patient?"
        "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
    )
}
input_str = [prompt_input.format_map(example)]

model_inputs = tokenizer(
    input_str,
    return_tensors='pt',
    padding=True,
)
print( f"\033[32mmodel_inputs\033[0m: { model_inputs }" )


topk_output = model.generate(
    model_inputs.input_ids.cuda(),
    max_new_tokens=1000,
    top_k=50
)
output_str = tokenizer.batch_decode(topk_output)
print('model predict: ', output_str[0])
```


## Training

The training process can be divided as two phases: pretrain and instruction-tuning.

**Pre-training**

The script for pretraining locates at `Pretrain/training.sh`.

Our pretraining dataset sources from [S2ORC](https://github.com/allenai/s2orc). Only those papers with PubMed IDs are deemed as medical-related and used during pretraining.
<!-- The raw training data can be dowloaded from [S2ORC](https://github.com/allenai/s2orc), filter out the papers with PubmedCentral IDs, and you can get the training data we use.  -->

The book is listed in this repo as [MedicalBook.xlsx](https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/MedicalBook.xlsx), due to licenses, we cannot release raw content. For reproducing, pls buy and process the books.

More details about how to fine-tune LLaMA can refer to [Finetune_LLAMA](https://github.com/chaoyi-wu/Finetune_LLAMA)


**Instruction Tuning**

We also provide instruction tuning script at `SFT/train.py`.
And you can find our instruction dataset at [PMC LLaMA Instructions](https://huggingface.co/datasets/axiong/pmc_llama_instructions).


## Results

### QA Benchmark
| Method              | Model Size          | USMLE | MedMCQA | PubMedQA |
|---------------------|---------------------|------------------|--------------|------------------|
| Human (pass)        | -                   | 50.0            | --            | 60.0           |
| Human (expert)      | -                   | 87.0            | 90.0         | 78.0           |
| ChatGPT             | 175B                | **57.0**        | 44.7         | 63.9           |
| LLaMA-2             | 13B                 | 42.73           | 37.41        | 68.0           |
| LLaMA-2             | 70B                 | 43.68           | 35.02        | 74.3           |
| Med-Alpaca          | 13B                 | 30.85           | 31.13        | 53.2           |
| Chat-Doctor         | 7B                  | 33.93           | 31.10        | 54.3           |
| PMC_LLaMA_13B ![](./figures/new.gif) | 13B | **56.36**   | **56.04**  | **77.9**  |


Note that, the manual and zero-shot results with * are referred from [LMFLow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow).


### Zero-shot Cases

We demonstrate PMC_LLaMA_13B's responses with out of domain queries.

<img src=./figures/pmc_llama_cases.png>

Note that, due to train on the papers, MedLLaMA_13B may generate some citation numbers (LLaMA somtimes will do this as well) and we dismiss them in the cases to show the main contents.
While for PMC_LLaMA_13B, it's much easier to extract the correct answer as the output result is structured.


## Acknowledge
Minimal LLaMA -- https://github.com/zphang/minimal-llama

alpaca -- https://github.com/tatsu-lab/stanford_alpaca

LMFLow -- https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

## Contact
If you have any question, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.

