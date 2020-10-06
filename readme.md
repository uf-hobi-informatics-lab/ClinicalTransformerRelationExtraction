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
export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=./new_model
pof=./predictions.txt
log=./log.txt

python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 2 \
		--pretrained_model bert-base-uncased \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_train \
		--do_predict \
		--do_lower_case \
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--learning_rate 1e-5 \
		--num_train_epochs 3 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \
```

- prediction
```shell script
export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=./new_model
pof=./predictions.txt
log=./log.txt

python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 2 \
		--pretrained_model bert-base-uncased \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_predict \
		--do_lower_case \
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--learning_rate 1e-5 \
		--num_train_epochs 3 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \
```

- post-processing (to brat format)
```shell script
# see --help for more information
data_dir=./sample_data
pof=./predictions.txt

python src/data_processing/post_processing.py \
		--mode mul \
		--predict_result_file $pof \
		--entity_data_dir ./test_data_entity_only \
		--test_data_file ${data_dir}/test.tsv \
		--brat_result_output_dir ./brat_output
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