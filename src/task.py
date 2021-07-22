"""
This script is used for training and test
"""


# from data_utils import convert_examples_to_relation_extraction_features
from data_utils import (features2tensors, relation_extraction_data_loader,
                        batch_to_model_input, RelationDataFormatSepProcessor,
                        RelationDataFormatUniProcessor)
from utils import acc_and_f1
from data_processing.io_utils import pkl_save, pkl_load
from transformers import glue_convert_examples_to_features as convert_examples_to_relation_extraction_features
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from tqdm import trange, tqdm
import numpy as np
from packaging import version
from pathlib import Path
from config import SPEC_TAGS, MODEL_DICT
import shutil


class TaskRunner(object):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model_dict = MODEL_DICT
        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None
        self.data_processor = None
        self._use_amp_for_fp16_from = 0

    def task_runner_default_init(self):
        # set up data processor
        if self.data_processor is None:
            if self.args.data_format_mode == 0:
                self.data_processor = RelationDataFormatSepProcessor(
                    max_seq_len=self.args.max_seq_length, num_core=self.args.num_core)
            elif self.args.data_format_mode == 1:
                self.data_processor = RelationDataFormatUniProcessor(
                    max_seq_len=self.args.max_seq_length, num_core=self.args.num_core)
            else:
                raise NotImplementedError("Only support 0, 1 but get data_format_mode as {}"
                                          .format(self.args.data_format_mode))
        else:
            self.args.logger.warning("Use user defined data processor: {}".format(self.data_processor))

        self.data_processor.set_data_dir(self.args.data_dir)
        self.data_processor.set_header(self.args.data_file_header)

        # init or reload model
        if self.args.do_train:
            # init amp for fp16 (mix precision training)
            # _use_amp_for_fp16_from: 0 for no fp16; 1 for naive PyTorch amp; 2 for apex amp
            if self.args.fp16:
                self._load_amp_for_fp16()
            self._init_new_model()
        else:
            self._init_trained_model()

        # load data
        self.data_processor.set_tokenizer(self.tokenizer)
        self.data_processor.set_tokenizer_type(self.args.model_type)
        self.args.logger.info("data loader info: {}".format(self.data_processor))
        self._init_dataloader()

        if self.args.do_train:
            self._init_optimizer()

        self.args.logger.info("Model Config:\n{}".format(self.config))
        self.args.logger.info("All parameters:\n{}".format(self.args))

    def train(self):
        # create data loader
        self.args.logger.info("start training...")
        tr_loss = .0
        t_step = 1
        latest_best_score = .0

        epoch_iter = trange(self.args.num_train_epochs, desc="Epoch", disable=not self.args.progress_bar)
        for epoch in epoch_iter:
            batch_iter = tqdm(self.train_data_loader, desc="Batch", disable=not self.args.progress_bar)
            batch_total_step = len(self.train_data_loader)
            for step, batch in enumerate(batch_iter):
                self.model.train()
                self.model.zero_grad()

                batch_input = batch_to_model_input(batch, model_type=self.args.model_type, device=self.args.device)

                if self.args.fp16 and self._use_amp_for_fp16_from == 1:
                    with self.amp.autocast():
                        batch_output = self.model(**batch_input)
                        loss = batch_output[0]
                else:
                    batch_output = self.model(**batch_input)
                    loss = batch_output[0]

                loss = loss / self.args.gradient_accumulation_steps
                tr_loss += loss.item()

                if self.args.fp16:
                    if self._use_amp_for_fp16_from == 1:
                        self.amp_scaler.scale(loss).backward()
                    elif self._use_amp_for_fp16_from == 2:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss.backward()

                # update gradient
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == batch_total_step:
                    if self.args.fp16:
                        if self._use_amp_for_fp16_from == 1:
                            self.amp_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.amp_scaler.step(self.optimizer)
                            self.amp_scaler.update()
                        elif self._use_amp_for_fp16_from == 2:
                            torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                           self.args.max_grad_norm)
                            self.optimizer.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                    if self.args.do_warmup:
                        self.scheduler.step()
                    # batch_iter.set_postfix({"loss": loss.item(), "tloss": tr_loss/step})
                if self.args.log_step > 0 and (step+1) % self.args.log_step == 0:
                    self.args.logger.info(
                        "epoch: {}; global step: {}; total loss: {}; average loss: {}".format(
                            epoch+1, t_step, tr_loss, tr_loss/t_step))

                t_step += 1
            batch_iter.close()

            # at each epoch end, we do eval on dev
            if self.args.do_eval:
                acc, prf, f1 = self.eval(self.args.non_relation_label)
                self.args.logger.info("""
                ******************************
                Epcoh: {}
                evaluation on dev set
                acc: {}
                precision, recall, f1:
                {}
                ******************************
                """.format(epoch+1, acc, prf))
                # max_num_checkpoints > 0, save based on eval
                # save model
                if self.args.max_num_checkpoints > 0 and latest_best_score < f1:
                    self._save_model(epoch+1)
                    latest_best_score = f1
        epoch_iter.close()

        # max_num_checkpoints=0 then save at the end of training
        if self.args.max_num_checkpoints <= 0:
            self._save_model(0)
            self.args.logger.info("training finish and the trained model is saved.")

    def eval(self, non_rel_label=""):
        self.args.logger.info("start evaluation...")

        # this is done on dev
        true_labels = np.array([dev_fea.label for dev_fea in self.dev_features])
        preds, eval_loss = self._run_eval(self.dev_data_loader)
        eval_res = acc_and_f1(
            labels=true_labels, preds=preds, label2idx=self.label2idx, non_rel_label=non_rel_label)

        return eval_res

    def predict(self):
        self.args.logger.info("start prediction...")
        # this is for prediction
        preds, _ = self._run_eval(self.test_data_loader)
        # convert predicted label idx to real label
        self.args.logger.info("label to index for prediction:\n{}".format(self.label2idx))
        preds = [self.idx2label[pred] for pred in preds]

        return preds

    def _init_new_model(self):
        """initialize a new model for fine-tuning"""
        self.args.logger.info("Init new model...")

        model, config, tokenizer = self.model_dict[self.args.model_type]

        # init tokenizer and add special tags
        self.tokenizer = tokenizer.from_pretrained(self.args.pretrained_model, do_lower_case=self.args.do_lower_case)
        last_token_idx = len(self.tokenizer)
        self.tokenizer.add_tokens(SPEC_TAGS)
        spec_token_new_ids = tuple([(last_token_idx + idx) for idx in range(len(self.tokenizer) - last_token_idx)])
        total_token_num = len(self.tokenizer)

        # init config
        unique_labels, label2idx, idx2label = self.data_processor.get_labels()
        self.args.logger.info("label to index:\n{}".format(label2idx))
        num_labels = len(unique_labels)
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.config = config.from_pretrained(self.args.pretrained_model, num_labels=num_labels)
        # The number of tokens to cache.
        # The key/value pairs that have already been pre-computed in a previous forward pass won’t be re-computed.
        if self.args.model_type == "xlnet":
            self.config.mem_len = self.config.d_model
            # change dropout name
            self.config.hidden_dropout_prob = self.config.dropout
        self.config.tags = spec_token_new_ids
        self.config.scheme = self.args.classification_scheme
        # binary mode
        self.config.binary_mode = self.args.use_binary_classification_mode
        # focal loss config
        self.config.use_focal_loss = self.args.use_focal_loss
        self.config.focal_loss_gamma = self.args.focal_loss_gamma
        # init model
        self.model = model.from_pretrained(self.args.pretrained_model, config=self.config)
        self.config.vocab_size = total_token_num
        self.model.resize_token_embeddings(total_token_num)

        # load model to device
        self.model.to(self.args.device)

    def _init_optimizer(self):
        # set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                           lr=self.args.learning_rate,
                                           eps=self.args.adam_epsilon)
        self.args.logger.info("The optimizer detail:\n {}".format(self.optimizer))

        # set up optimizer warm up scheduler (you can set warmup_ratio=0 to deactivated this function)
        if self.args.do_warmup:
            t_total = len(self.train_data_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
            warmup_steps = np.dtype('int64').type(self.args.warmup_ratio * t_total)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=warmup_steps,
                                                             num_training_steps=t_total)

        # mix precision training
        if self.args.fp16 and self._use_amp_for_fp16_from == 2:
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer,
                                                             opt_level=self.args.fp16_opt_level)

    def _init_trained_model(self):
        """initialize a fine-tuned model for prediction"""
        p = Path(self.args.new_model_dir)
        dir_list = [d for d in p.iterdir() if d.is_dir()]
        latest_ckpt_dir = sorted(dir_list, key=lambda x: int(x.stem.split("_")[-1]))[-1]

        self.args.logger.info("Init model from {} for prediction".format(latest_ckpt_dir))

        model, config, tokenizer = self.model_dict[self.args.model_type]

        self.config = config.from_pretrained(latest_ckpt_dir)
        self.tokenizer = tokenizer.from_pretrained(latest_ckpt_dir, do_lower_case=self.args.do_lower_case)
        self.model = model.from_pretrained(latest_ckpt_dir, config=self.config)

        # load label2idx
        self.label2idx, self.idx2label = pkl_load(latest_ckpt_dir/"label_index.pkl")
        # load model to device
        self.model.to(self.args.device)

    def _load_amp_for_fp16(self):
        # first try to load PyTorch naive amp; if fail, try apex; if fail again, throw a RuntimeError
        if version.parse(torch.__version__) >= version.parse("1.6.0"):
            self.amp = torch.cuda.amp
            self._use_amp_for_fp16_from = 1
            self.amp_scaler = torch.cuda.amp.GradScaler()
        else:
            try:
                from apex import amp
                self.amp = amp
                self._use_amp_for_fp16_from = 2
            except ImportError:
                self.args.logger.error("apex (https://www.github.com/nvidia/apex) for fp16 training is not installed.")
            finally:
                self.args.fp16 = False

    def _save_model(self, epoch=0):
        p = Path(self.args.new_model_dir)
        p.mkdir(parents=True, exist_ok=True)
        dir_to_save = p / f"ckpt_{epoch}"

        self.tokenizer.save_pretrained(dir_to_save)
        self.config.save_pretrained(dir_to_save)
        self.model.save_pretrained(dir_to_save)
        # save label2idx
        pkl_save((self.label2idx, self.idx2label), dir_to_save/"label_index.pkl")
        # remove extra checkpoints
        dir_list = [d for d in p.iterdir() if d.is_dir()]
        if len(dir_list) > self.args.max_num_checkpoints > 0:
            oldest_ckpt_dir = sorted(dir_list, key=lambda x: int(x.stem.split("_")[-1]))[0]
            shutil.rmtree(oldest_ckpt_dir)

    def _run_eval(self, data_loader):
        temp_loss = .0
        # set model to evaluate mode
        self.model.eval()

        # create dev data batch iteration
        batch_iter = tqdm(data_loader, desc="Batch", disable=not self.args.progress_bar)
        total_sample_num = len(batch_iter)
        preds = None

        for batch in batch_iter:
            batch_input = batch_to_model_input(batch, model_type=self.args.model_type, device=self.args.device)
            with torch.no_grad():
                batch_output = self.model(**batch_input)
                loss, logits = batch_output[:2]
                temp_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                preds = logits if preds is None else np.append(preds, logits, axis=0)

        batch_iter.close()
        temp_loss = temp_loss / total_sample_num
        preds = np.argmax(preds, axis=-1)

        return preds, temp_loss

    def _load_examples_by_task(self, task="train"):
        examples = None

        if task == "train":
            examples = self.data_processor.get_train_examples()
        elif task == "dev":
            examples = self.data_processor.get_dev_examples()
        elif task == "test":
            examples = self.data_processor.get_test_examples()
        else:
            raise RuntimeError("expect task to be train, dev or test but get {}".format(task))

        return examples

    def _check_cache(self, task="train"):
        cached_examples_file = Path(self.args.data_dir) / "cached_{}_{}_{}_{}_{}.pkl".format(
            self.args.model_type, self.args.data_format_mode, self.args.max_seq_length,
            self.tokenizer.name_or_path.split("/")[-1], task)
        # load examples from files or cache
        if self.args.cache_data and cached_examples_file.exists():
            examples = pkl_load(cached_examples_file)
            self.args.logger.info("load {} data from cached file: {}".format(task, cached_examples_file))
        elif self.args.cache_data and not cached_examples_file.exists():
            self.args.logger.info(
                "create {} examples...and will cache the processed data at {}".format(task, cached_examples_file))
            examples = self._load_examples_by_task(task)
            pkl_save(examples, cached_examples_file)
        else:
            self.args.logger.info("create training examples..."
                                  "the processed data will not be cached")
            examples = self._load_examples_by_task(task)
        return examples

    def _init_dataloader(self):
        if self.args.do_train and self.train_data_loader is None:
            train_examples = self._check_cache(task="train")
            # convert examples to tensor
            train_features = convert_examples_to_relation_extraction_features(
                train_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification")

            self.train_data_loader = relation_extraction_data_loader(
                train_features,
                batch_size=self.args.train_batch_size,
                task="train",
                logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode)

        if self.args.do_eval and self.dev_data_loader is None:
            dev_examples = self._check_cache(task="dev")
            # example2feature
            dev_features = convert_examples_to_relation_extraction_features(
                dev_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification")
            self.dev_features = dev_features

            self.dev_data_loader = relation_extraction_data_loader(
                dev_features,
                batch_size=self.args.train_batch_size,
                task="test",
                logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode)

        if self.args.do_predict and self.test_data_loader is None:
            test_examples = self._check_cache(task="test")
            # example2feature
            test_features = convert_examples_to_relation_extraction_features(
                test_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification")

            self.test_data_loader = relation_extraction_data_loader(
                test_features,
                batch_size=self.args.eval_batch_size,
                task="test", logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode)
