# example of training using json config to initialize all experiment parameters
export CUDA_VISIBLE_DEVICES=1

python ./src/relation_extraction_json.py \
		--config_json "./config_experiment_sample.json"