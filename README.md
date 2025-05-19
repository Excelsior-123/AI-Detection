## Research on Domain Generalization Methods for AI Text Detection

This repository contains the code and data for my graduation project.

## Overview

- Contrastive Learning Framework: Proposed a contrastive learning-based cross-domain text classification framework using a shared-weight RoBERTa-OpenAI encoder for consistent sentence-level semantic representations.
- Dual Loss Design: Designed a joint loss combining contrastive loss (cosine similarity) and binary cross-entropy to improve both representation distinctiveness and classification capability.
- End-to-End Fine-Tuning for Transfer Learning: Performed end-to-end fine-tuning on the RAID benchmark dataset, including both the pretrained encoder and the classification head.
- Comprehensive Evaluation: Evaluated performance using metrics such as Accuracy, Precision, Recall, F1-score, AUROC, STD and Domain Gap compared against baseline models.
- Cross-Domain Effectiveness: Achieved significant improvements in cross-domain AI text detection, demonstrating strong generalization and domain transfer capabilities.

![](figure/model.png)

## Getting Started

The evaluation contains a baseline using by SimCSE, first install the `simcse` package from PyPI, you can also refer to the link [SimCSE](https://github.com/princeton-nlp/SimCSE)

```bash
pip install simcse
```

Note that if you want to enable GPU encoding, you should install the correct version of PyTorch that supports CUDA. See [PyTorch official website](https://pytorch.org) for instructions.

## Model List

You can download the models using [HuggingFace's Transformers](https://github.com/huggingface/transformers). 


### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Datasets
You can download the datasets via baidu network disk. I will release them later.

### Training & Evaluation
Adjust the pre-trained model and the tokenizer via "simcse/models.py" and then run "train.py" for training. Note that you should offer the arguments like "--training_file", "--model_name_or_path", etc.
If you use our method, please refer to "SemEval24-task8/src/train.py", remember to adjust the arguments via "base_config.py". You can later transfer the ".pt" weight to the HF model for the downstream fine-tune tasks.
For the downstream fine-tune task, only run "train_ours.py". It will automatically evaluate the model using the testing and evaluating datasets.

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Liangyan (`liliangyan_21_ustb@163.com`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!
