#Clinical Relation Extration with Transformers

## Aim
This package is developed for researchers easily to use state-of-the-art transformers models for extracting relations from clinical notes. 
No prior knowledge of transformers is required. We handle the whole process from data preprocessing to training to prediction.

## Dependency
The package is built on top of the Transformers developed by the HuggingFace. 
We have the requirement.txt to specify the packages required to run the project.

## Background
Our training strategy is inspired by the paper: https://arxiv.org/abs/1906.03158

## Available models
- BERT
- XLNet
- RoBERTa
- ABERT
> We will keep adding new models.

## usage and example
We require the data must have original raw text and entities. 
We will generate candidate relations as entity pairs. 
We will use brat annotation format as default.

- preprocess data

- training (use 5-CV for hyperparameter optimization)

- prediction

## Issues
raise an issue if you have problems. 

## Citation
please cite our paper:
```

```

## Clinical Pre-trained Transformer Models
We have a series transformer models pre-trained on MIMIC-III.
You can find them here:
- https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip