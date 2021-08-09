#!/bin/bash

# example for how to run prediction on batch data
# see ./src/data_processing/preprocessing-batch.ipynb for example on how to generate batch data for prediction
# we assume the batch data is in ./data/batch_data
# you should expect there are several directories named as batch_* with a test.tsv file in each of them

export CUDA_VISIBLE_DEVICES=1

model_type=bert
data_format_mode=0
max_seq_length=512
data_file_has_header=True
tag_for_non_relation=NonRel
mode=bin # binary tags pos/neg
binary_type_mapping_file=./data/mappings.json
relation_extraction_model=./model
data_dir=./data/batch_data
pred_dir=./data/predict_batch
brat_entity_dir=./data/results_from_NER
final_brat_output_with_NER_RE=./data/final
log=./log.txt

python ./src/batch_prediction.py \
  --model_type $model_type \
  --data_format_mode $data_format_mode \
  --new_model_dir $relation_extraction_model \
  --predict_output_dir $pred_dir \
  --max_seq_length $max_seq_length \
  --data_file_header $data_file_has_header \
  --do_lower_case \
  --eval_batch_size 32 \
  --log_file $log \
  --log_lvl i \
  --num_core 4 \
  --non_relation_label $tag_for_non_relation \
  --classification_mode $mode \
  --type_map $binary_type_mapping_file \
  --entity_data_dir $brat_entity_dir \
  --brat_result_output_dir $final_brat_output_with_NER_RE

