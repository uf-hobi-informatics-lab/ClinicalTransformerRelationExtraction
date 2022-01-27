"""
The py file is an example for

1. how to run RE as an python app not through command line
2. how to manually add new models other than those available in the models.py

We used deberta as an example
"""
from models import BaseModel
from data_utils import RelationDataFormatSepProcessor
from transformers import DebertaForSequenceClassification, DebertaModel, DebertaConfig, DebertaTokenizer
from task import TaskRunner
from utils import TransformerLogger

import numpy as np
import torch
import traceback


class DeBERTaRelationExtraction(DebertaForSequenceClassification, BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_output = outputs[0]
        pooled_output = self.pooler(seq_output)

        pooled_output = self.dropout(pooled_output)
        seq_output = self.dropout(seq_output)

        logits = self.output2logits(pooled_output, seq_output, input_ids)
        outputs = (logits,) + outputs[2:]
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs


class DeBERTaDataProcessor(RelationDataFormatSepProcessor):
    def __init__(self, data_dir=None, max_seq_len=128, num_core=-1, header=True, tokenizer_type='deberta'):
        super().__init__(
            data_dir=data_dir, max_seq_len=max_seq_len, num_core=num_core, header=True, tokenizer_type='deberta')
        self.total_special_token_num = 4


class Args:
    """
        used to init all parameters
        deberta use roberta vocab
        deberta-v2 need new tokenizer as XLNet based on SPM
    """
    def __init__(self, **kwargs):
        self.model_type = "deberta"
        self.data_format_mode = 0
        self.classification_scheme = 2
        self.pretrained_model = "microsoft/deberta-base"  # microsoft/deberta-large; microsoft/deberta-xlarge
        self.data_dir = "../sample_data"
        self.new_model_dir = "../temp/deberta_re_model"
        self.predict_output_file = "../temp/deberta_re_predict.txt"
        self.overwrite_model_dir = True
        self.seed = 1234
        self.max_seq_length = 128
        self.cache_data = False
        self.data_file_header = True
        self.do_train = True
        self.do_eval = True
        self.do_predict = True
        self.do_lower_case = True
        self.train_batch_size = 2
        self.eval_batch_size = 32
        self.learning_rate = 1e-5
        self.num_train_epochs = 5
        self.gradient_accumulation_steps = 1
        self.do_warmup = True
        self.warmup_ratio = 0.1
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_num_checkpoints = 0
        self.log_file = None
        self.log_lvl = "i"
        self.log_step = 2
        self.num_core = 4
        self.non_relation_label = "nonRel"
        self.progress_bar = False
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


def app():
    args = Args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = TransformerLogger(logger_file=args.log_file, logger_level='i').get_logger()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_runner = TaskRunner(args)

    # add deberta to model dict
    task_runner.model_dict['deberta'] = (DeBERTaRelationExtraction, DebertaConfig, DebertaTokenizer)
    # set deberta data processor for data processing
    task_runner.data_processor = DeBERTaDataProcessor(
        max_seq_len=args.max_seq_length, num_core=args.num_core)

    task_runner.task_runner_default_init()

    if args.do_train:
        try:
            task_runner.train()
        except Exception as ex:
            raise RuntimeError(traceback.print_exc())

    if args.do_predict:
        try:
            task_runner.predict()
        except Exception as ex:
            raise RuntimeError(traceback.print_exc())


if __name__ == '__main__':
    app()
