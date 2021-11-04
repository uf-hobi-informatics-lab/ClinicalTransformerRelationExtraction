# Clinical Relation Extration with Transformers

## Aim
This package is developed for researchers easily to use state-of-the-art transformers models for extracting relations from clinical notes. 
No prior knowledge of transformers is required. We handle the whole process from data preprocessing to training to prediction.

## Dependency
The package is built on top of the Transformers developed by the HuggingFace. 
We have the requirement.txt to specify the packages required to run the project.

## Background
Our training strategy is inspired by the paper: https://arxiv.org/abs/1906.03158
We only support train-dev mode, but you can do 5-fold CV.

## Available models
- BERT
- XLNet
- RoBERTa
- ALBERT
- DeBERTa
- Longformer
> We will keep adding new models.

## usage and example
- prerequisite
> The package is only for relation extraction, thus the entities must be provided. 
> You have to conduction NER first to get all entities then run this package to get the end-to-end relation extraction results

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

note: 
1) the entity between [s1][e1] is the first entity in a relation; the second entity in the relation is inbetween [s2][e2]
2) even the two entities in the same sentenc, we still require to put them separately
3) in the test.tsv, you can set all labels to neg or no_relation or whatever, because we will not use the label anyway
4) We recommend to evaluate the test performance in a separate process based on prediction. (see **post-processing**)
5) We recommend using official evaluation scripts to do evaluation to make sure the results reported are reliable.
```

- preprocess data (see the preprocess.ipynb script for more details on usage)
> we did not provide a script for training and test data generation

> we have a jupyter notebook with preprocessing 2018 n2c2 data as an example

> you can follow our example to generate your own dataset

- special tags
> we use 4 special tags to identify two entities in a relation
```
# the defaults tags we defined in the repo are

EN1_START = "[s1]"
EN1_END = "[e1]"
EN2_START = "[s2]"
EN2_END = "[e2]"

If you need to customize these tags, you can change them in
config.py
```

- training
> please refer to the wiki page for all details of the parameters
> [flag details](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction/wiki/all-parameters)

```shell script
export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=./new_modelzw
pof=./predictions.txt
log=./log.txt

# NOTE: we have more options available, you can check our wiki for more information
python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model bert-base-uncased \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_train \
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

# we have to set data_dir, new_model_dir, model_type, log_file, and eval_batch_size, data_format_mode
python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 1 \
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
		--eval_batch_size 4 \
		--log_file $log \
```

- post-processing (we only support transformation to brat format)
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


## Using json file for experiment config instead of commend line

- to simplify using the package, we support using json file for configuration
- using json, you can define all parameters in a separate json file instead of input via commend line
- config_experiment_sample.json is a sample json file you can follow to develop yours
- to run experiment with json config, you need to follow run_json.sh
```shell script
export CUDA_VISIBLE_DEVICES=1

python ./src/relation_extraction_json.py \
		--config_json "./config_experiment_sample.json"
```

## Inference on a large corpus
- If you have a model and need to run inference on a large corpus, we can refer to **batch_prediction.py**
- We also have the preprocessing notebook for batch_prediction.py in /data_preprocessing

## Baseline (baseline directory)
- We also implemented some baselines for relation extraction using machine learning approaches
- baseline is for comparison only
- baseline based on SVM
- features extracted may not optimize for each dataset (cover most commonly used lexical and semantic features)
- see baseline/run.sh for example

## Issues
raise an issue if you have problems. 

## Citation
please cite our paper:
```
# We have a preprint at
https://arxiv.org/abs/2107.08957
```

## Clinical Pre-trained Transformer Models
We have a series transformer models pre-trained on MIMIC-III.
You can find them here:
- https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz
- https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip
