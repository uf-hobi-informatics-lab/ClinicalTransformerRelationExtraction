import argparse
import torch
import numpy as np
import random
from utils import TransformerLogger
from task import TaskRunner
from pathlib import Path
from data_processing.io_utils import save_text, save_json
import traceback
import warnings


def set_seed(gargs):
    random.seed(gargs.seed)
    np.random.seed(gargs.seed)
    torch.manual_seed(gargs.seed)


def check_args(args):
    # do_eval is used with do_train in most cases for 5-CV
    if args.do_eval and not args.do_train:
        raise RuntimeError("Evaluation mode (do_eval) is only available when do_train is used.\n"
                           "You may want to use do_predict instead.")

    if args.max_num_checkpoints > 0 and not args.do_eval:
        warnings.warn("Evaluation mode (do_eval) should be set in order to save more than one models."
                        "We will evaluate at the end of each epoch and save models with better F1-score."
                        "if do_eval is not set, we will only save one model at the end of training,"
                        "in this case you have to set max_num_checkpoints=0 (default),"
                        "We did this for you by setting max_num_checkpoints=0"
                      )
        args.max_num_checkpoints = 0

    if args.do_eval and args.max_num_checkpoints < 1:
        warnings.warn("You set the eval mode so we expect max_num_checkpoints large than 0 so we set it to 1.")
        args.max_num_checkpoints = 1

    if args.do_train and Path(args.new_model_dir).exists() and not args.overwrite_model_dir:
        raise RuntimeError("{} is exist and overwrite this dir is not permitted.".format(args.new_model_dir))

    if args.use_binary_classification_mode:
        line = "*" * 20
        info = "You turn on the binary mode, make sure you use binary data format."
        warnings.warn(f"{line}\n{info}\n{line}\n")


def app(gargs):
    set_seed(gargs)
    check_args(gargs)

    # make model type case in-sensitive
    gargs.model_type = gargs.model_type.lower()
    task_runner = TaskRunner(gargs)
    task_runner.task_runner_default_init()

    if gargs.do_train:
        # training
        try:
            task_runner.train()
        except Exception as ex:
            gargs.logger.error("Training error:\n{}".format(traceback.format_exc()))
            traceback.print_exc()
            raise RuntimeError()

    if gargs.do_predict:
        # run prediction
        try:
            preds = task_runner.predict()
        except Exception as ex:
            gargs.logger.error("Prediction error:\n{}".format(traceback.format_exc()))
            raise RuntimeError(traceback.format_exc())

        pred_res = "\n".join([str(pred) for pred in preds])

        # predict_output_file must be a file, we will create parent dir automatically
        Path(gargs.predict_output_file).parent.mkdir(parents=True, exist_ok=True)
        save_text(pred_res, gargs.predict_output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="valid values: bert, roberta, albert, xlnet, megatron, deberta, longformer")
    parser.add_argument("--data_format_mode", default=0, type=int,
                        help="valid values: 0: sep mode - [CLS]S1[SEP]S2[SEP]; 1: uni mode - [CLS]S1S2[SEP]")
    parser.add_argument("--classification_scheme", default=2, type=int,
                        help="special tokens used for classification. "
                             "Valid values: "
                             "0: [CLS]; 1: [CLS], [S1], [S2]; 2: [CLS], [S1], [E1], [S2], [E2]; 3: [S1], [S2]")
    parser.add_argument("--pretrained_model", type=str,
                        help="The pretrained model file or directory for fine tuning.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory. Should have at least a file named train.tsv")
    parser.add_argument("--new_model_dir", type=str, required=True,
                        help="directory for saving new model checkpoints (keep latest n only)")
    parser.add_argument("--predict_output_file", type=str, default=None,
                        help="predicted results output file.")
    parser.add_argument('--overwrite_model_dir', action='store_true',
                        help="Overwrite the content of the new model directory")
    parser.add_argument("--seed", default=1234, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--cache_data", action='store_true',
                        help="Whether to cache the features after tokenization (save training initialization time)")
    parser.add_argument("--data_file_header", default=True, type=bool,
                        help="flag used to define whether the data tsv file has header or not. "
                             "If has header, we will skip the first line")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation on dev. (require dev.tsv)")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run prediction on the test set. (require test.tsv)")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="The batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="The batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_warmup", action='store_true',
                        help='Whether to apply warmup strategy in optimizer.')
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_num_checkpoints", default=0, type=int,
                        help="max number of checkpoints saved during training, old checkpoints will be removed."
                             "if 0, then only save the last one at the end of training")
    parser.add_argument("--log_file", default=None,
                        help="where to save the log information")
    parser.add_argument("--log_lvl", default="i", type=str,
                        help="d=DEBUG; i=INFO; w=WARNING; e=ERROR")
    parser.add_argument("--log_step", default=1000, type=int,
                        help="logging after how many steps of training. If < 0, no log during training")
    parser.add_argument("--num_core", default=1, type=int,
                        help="how many cores used for multiple process for data generation")
    parser.add_argument("--non_relation_label", default="NonRel", type=str,
                        help="The label used for representing "
                             "candidate entity pairs that is not a true relation (negative sample)")
    parser.add_argument("--progress_bar", action='store_true',
                        help="show progress during the training in tqdm")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--use_focal_loss', action='store_true',
                        help="Whether to use focal loss function to replace cross entropy loss function")
    parser.add_argument("--focal_loss_gamma", default=2, type=int,
                        help="focussing parameter used in focal loss function")
    parser.add_argument('--use_binary_classification_mode', action='store_true',
                        help="if use this mode, we will use BCEWithLogitsLoss or binary focal loss functions.")
    parser.add_argument('--balance_sample_weights', action='store_true',
                        help="Whether to create sample weights and pass it to loss functions")
    # using pytorch ddp
    # parser.add_argument('--ddp', action='store_true',
    #                     help="Whether to use Distributed Data Parallel")
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help="local rank ID")

    args = parser.parse_args()
    # save the experiment arguments into a file under new model dir
    Path(args.new_model_dir).mkdir(exist_ok=True, parents=True)
    save_json(vars(args), Path(args.new_model_dir) / "training_arguments.json")

    # other setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = TransformerLogger(logger_file=args.log_file, logger_level=args.log_lvl).get_logger()
    app(args)
