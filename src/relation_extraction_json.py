import argparse
import json
import torch
from utils import TransformerLogger
from relation_extraction import app as main_app


class Args:
    """
        used to hold all parameters
        actual parameters for experiments will be loaded from the user defined json config file
    """
    def __init__(self, **kwargs):
        self.model_type = "bert"
        self.data_format_mode = 0
        self.classification_scheme = 2
        self.pretrained_model = "bert-base-uncased" # microsoft/deberta-large; microsoft/deberta-xlarge
        self.data_dir = "../sample_data"
        self.new_model_dir = "./bert_re_model"
        self.predict_output_file = "./bert_re_predict.txt"
        self.overwrite_model_dir = True
        self.seed = 1234
        self.max_seq_length = 128
        self.cache_data = False
        self.data_file_header = True
        self.do_train = True
        self.do_eval = False
        self.do_predict = True
        self.do_lower_case = True
        self.train_batch_size = 8
        self.eval_batch_size = 32
        self.learning_rate = 1e-5
        self.num_train_epochs = 4
        self.gradient_accumulation_steps = 1
        self.do_warmup = True
        self.warmup_ratio = 0.1
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_num_checkpoints = 0
        self.log_file = "./bert_re_log_txt"
        self.log_lvl = "i"
        self.log_step = 100
        self.num_core = 4
        self.non_relation_label = "nonRel"
        self.progress_bar = True
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.use_focal_loss = False
        self.focal_loss_gamma = 2
        self.use_binary_classification_mode = False
        self.balance_sample_weights = False

        self.__update_args(**kwargs)

    def __update_args(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return repr(self.__dict__)


def json2args(jsondata):
    return Args(**jsondata)


def app(gargs):
    main_app(gargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--config_json", default="./config.json", type=str, required=True,
                        help="json file for experiment configurations")
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        configs = json.load(f, object_hook=json2args)

    # other setup
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.logger = TransformerLogger(
        logger_file=configs.log_file, logger_level=configs.log_lvl).get_logger()

    app(configs)
