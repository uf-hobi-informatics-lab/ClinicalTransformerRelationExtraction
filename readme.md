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
- data format
> see sample_data dir (train.tsv and test.tsv) for the train and test data format

> The sample data is a small subset of the data prepared from the 2018 umass made1.0 challenge corpus

```
# data format: tsv file with 8 columns:
1. relation_type: adverse
2. sentence_1: ALLERGIES : [s1] Penicillin [e1] .
3. sentence_2: [s2] ALLERGIES [e2] : Penicillin .
4. entity_type_1: Drug
5. entity_type_2: ADE
6. entity_id_1: T1
7. entity_id2: T2
8. file_id: 13_10
```

- preprocess data (see the preprocess.ipynb script for more details on usage)
> we did not provide a script for training and test data generation

> we have a jupyter notebook with preprocessing 2018 n2c2 data as an example

> you can follow our example to generate your own dataset

- training
```shell script

```

- prediction
```shell script

```

- post-processing (to brat format)
```shell script

```

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