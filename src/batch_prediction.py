# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 8/5/21

"""
Here we aim to solve the issue when we have thousands to millions nodes to deal with
The relation_extraction.py is designed for small corpus, but in reality, we may need to due thousands of nodes
See /data_processing/preprocessing_batch.ipynb for how to prepare input data
"""


import argparse
import distutils

import torch
from utils import TransformerLogger
from task import TaskRunner
from pathlib import Path
from data_processing.io_utils import save_text
import traceback
import warnings
from data_utils import (features2tensors, relation_extraction_data_loader,
                        batch_to_model_input, RelationDataFormatSepProcessor,
                        RelationDataFormatUniProcessor)
from data_processing.post_processing import app as post_processing


class BatchRunner(TaskRunner):
    def task_runner_batch_init(self):
        # set up data processor
        if self.args.data_format_mode == 0:
            self.data_processor = RelationDataFormatSepProcessor(
                max_seq_len=self.args.max_seq_length, num_core=self.args.num_core)
        elif self.args.data_format_mode == 1:
            self.data_processor = RelationDataFormatUniProcessor(
                max_seq_len=self.args.max_seq_length, num_core=self.args.num_core)
        else:
            raise NotImplementedError("Only support 0, 1 but get data_format_mode as {}"
                                      .format(self.args.data_format_mode))

        self._init_trained_model()
        self.data_processor.set_tokenizer(self.tokenizer)
        self.data_processor.set_tokenizer_type(self.args.model_type)


def app(gargs):
    # make model type case in-sensitive
    gargs.model_type = gargs.model_type.lower()
    gargs.progress_bar = False
    gargs.cache_data = False

    task_runner = BatchRunner(gargs)
    # no data loader init, we init data loader in reset function during the loop
    task_runner.task_runner_batch_init()

    for batch_id, each_batch_dir in enumerate(Path(gargs.data_dir).iterdir()):
        try:
            task_runner.reset_dataloader(each_batch_dir,
                                         has_file_header=gargs.data_file_header,
                                         max_len=gargs.max_seq_length)
            gargs.logger.info("data loader info: {}".format(task_runner.data_processor))
            preds = task_runner.predict()
        except Exception as ex:
            gargs.logger.error("Prediction error:\n{}".format(traceback.format_exc()))
            raise RuntimeError(traceback.format_exc())

        pred_res = "\n".join([str(pred) for pred in preds])

        # predict_output_file must be a file, we will create parent dir automatically
        p_pred = Path(gargs.predict_output_dir)
        p_pred.mkdir(parents=True, exist_ok=True)
        save_text(pred_res, gargs.predict_output_dir/f"batch_{batch_id}_prediction.txt")

        # output to files
        gargs.mode = gargs.classification_mode
        gargs.neg_type = gargs.non_relation_label
        gargs.predict_result_file = gargs.predict_output_dir/f"batch_{batch_id}_prediction.txt"
        gargs.test_data_file = each_batch_dir / "test.tsv"
        post_processing(gargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="valid values: bert, roberta, albert, xlnet, megatron, deberta, longformer")
    parser.add_argument("--data_format_mode", default=0, type=int,
                        help="valid values: 0: sep mode - [CLS]S1[SEP]S2[SEP]; 1: uni mode - [CLS]S1S2[SEP]")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory. Should have at least a file named train.tsv")
    parser.add_argument("--new_model_dir", type=str, required=True,
                        help="directory for saving new model checkpoints (keep latest n only)")
    parser.add_argument("--predict_output_dir", type=str, default=None,
                        help="predicted results output file.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--data_file_header", default=True, type=bool,
                        help="flag used to define whether the data tsv file has header or not. "
                             "If has header, we will skip the first line")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="The batch size for eval.")
    parser.add_argument("--log_file", default=None,
                        help="where to save the log information")
    parser.add_argument("--log_lvl", default="i", type=str,
                        help="d=DEBUG; i=INFO; w=WARNING; e=ERROR")
    parser.add_argument("--num_core", default=1, type=int,
                        help="how many cores used for multiple process for data generation")
    parser.add_argument("--non_relation_label", default="NonRel", type=str,
                        help="The label used for representing "
                             "candidate entity pairs that is not a true relation (negative sample)")
    parser.add_argument("--classification_mode", type=str, default='mul', required=True,
                        help="we have two mode for binary (bin) and multiple (mul) classes classification")
    parser.add_argument("--type_map", type=str, default=None,
                        help="a map of entity pair types to relation types (only use when mode is bin)")
    parser.add_argument("--entity_data_dir", type=str, required=True,
                        help="The annotation/NER output files with only the entities. Used for output NER and RE.")
    parser.add_argument("--brat_result_output_dir", type=str, required=True,
                        help="prediction results")

    args = parser.parse_args()

    # other setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = TransformerLogger(logger_file=args.log_file, logger_level=args.log_lvl).get_logger()
    app(args)
