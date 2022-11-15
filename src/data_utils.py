import traceback
from .config import MODEL_REQUIRE_SEGMENT_ID, SPEC_TAGS, TOKENIZER_USE_FOUR_SPECIAL_TOKs
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import re
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import Counter


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The not tokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The not tokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}={}\n".format(k, v)
        return s


def convert_examples_to_relation_extraction_features(
        examples, label2idx, tokenizer, max_length=128):
    """This function is the same as transformers.glue_convert_examples_to_features"""
    features = []

    for idx, example in enumerate(tqdm(examples)):
        text_a, text_b = example.text_a, example.text_b

        tokens_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_a))

        if text_b:
            tokens_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b))
        else:
            tokens_b = None

        inputs = tokenizer.encode_plus(
            tokens_a, tokens_b, pad_to_max_length=True, max_length=max_length, truncation=False)

        label = label2idx[example.label]
        feature = InputFeatures(**inputs, label=label)
        features.append(feature)

        if idx < 3:
            print("###exampel###\nguide: {}\ntext: {}\ntoken ids: {}\nmasks: {}\nlabel: {}\n########".format(
                example.guid,
                example.text_a + " " + example.text_b,
                feature.input_ids,
                feature.attention_mask,
                feature.label))

    return features


def features2tensors(features, binary_mode=False, logger=None):
    tensor_input_ids = []
    tensor_attention_masks = []
    tensor_token_type_ids = []
    tensor_label_ids = []

    for idx, feature in enumerate(features):
        if logger and idx < 3:
            logger.info("Feature{}:\n{}\n".format(idx + 1, feature))

        tensor_input_ids.append(feature.input_ids)
        tensor_attention_masks.append(feature.attention_mask)
        tensor_label_ids.append(feature.label)

        if feature.token_type_ids:
            tensor_token_type_ids.append(feature.token_type_ids)

    tensor_input_ids = torch.tensor(tensor_input_ids, dtype=torch.long)
    tensor_attention_masks = torch.tensor(tensor_attention_masks, dtype=torch.long)
    tensor_token_type_ids = torch.tensor(tensor_token_type_ids, dtype=torch.long) if tensor_token_type_ids \
        else torch.zeros(tensor_attention_masks.shape)
    if binary_mode:
        dmap = {0: [1, 0], 1: [0, 1]}
        tensor_label_ids = torch.tensor([dmap[e] for e in tensor_label_ids], dtype=torch.float32)
    else:
        tensor_label_ids = torch.tensor(tensor_label_ids, dtype=torch.long)

    return TensorDataset(tensor_input_ids, tensor_attention_masks, tensor_token_type_ids, tensor_label_ids)


def relation_extraction_data_loader(dataset, batch_size=2, task='train', logger=None, binary_mode=False):
    """
    task has two levels:
    train for training using RandomSampler
    test for evaluation and prediction using SequentialSampler

    if set auto to True we will default call convert_features_to_tensors,
    so features can be directly passed into the function
    """
    dataset = features2tensors(dataset, binary_mode=binary_mode, logger=logger)

    if task == 'train':
        sampler = RandomSampler(dataset)
    elif task == 'test':
        sampler = SequentialSampler(dataset)
    else:
        raise ValueError('task argument only support train or test but get {}'.format(task))

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)

    return data_loader


def batch_to_model_input(batch, model_type="bert", device=torch.device("cpu")):
    return {"input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "labels": batch[3].to(device),
            "token_type_ids": batch[2].to(device) if model_type in MODEL_REQUIRE_SEGMENT_ID else None}


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir=None, max_seq_len=128, num_core=1, header=True, tokenizer_type='bert'):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = data_dir

        self.tokenizer = None
        self.max_seq_len = max_seq_len
        self.num_core = num_core
        self.header = header
        self.tokenizer_type = tokenizer_type
        self.total_special_token_num = 3

    def __str__(self):
        rep = [f"key: {k}; val: {v}" for k, v in self.__dict__.items()]
        return "\n".join(rep)

    def set_data_dir(self, data_dir):
        self.data_dir = Path(data_dir)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def set_tokenizer_type(self, tokenizer_type):
        self.tokenizer_type = tokenizer_type

    def set_num_core(self, num_core):
        self.num_core = num_core

    def set_header(self, header):
        self.header = header

    def get_train_examples(self, filename=None):
        """See base class."""
        input_file_name = self.data_dir / filename if filename else self.data_dir / "train.tsv"

        return self._create_examples(
            self._read_tsv(input_file_name), "train")

    def get_dev_examples(self, filename=None):
        """See base class."""
        input_file_name = self.data_dir / filename if filename else self.data_dir / "dev.tsv"

        return self._create_examples(
            self._read_tsv(input_file_name), "dev")

    def get_test_examples(self, filename=None, tsv=None):
        """See base class."""
        input_file_name = self.data_dir / filename if filename else self.data_dir / "test.tsv"

        return self._create_examples(
            (self._read_tsv(input_file_name) if (tsv is None) else tsv), "test")

    def get_sample_distribution(self, train_file=None):
        # the distribution will be measured based on training data
        if train_file:
            lines = self._read_tsv(train_file)
        else:
            lines = self._read_tsv(self.data_dir / "train.tsv")

        labels = []
        for (i, line) in enumerate(lines):
            labels.append(line[0])
        total = len(labels)
        label2freq = {k: (1-v/total) for k, v in Counter(labels).items()}

        return label2freq

    def get_labels(self, train_file=None, label_file=None):
        """
            Gets the list of labels for this data set.
            1. use labels in train file for indexing
                In all different formats, the first column always should be label
            2. add a label index file
                A plain text with each unique label in one line
        """
        if label_file:
            with open(label_file, "r") as f:
                unique_labels = [e.strip() for e in f.read().strip().split("\n")]
        elif label_file is None and train_file:
            lines = self._read_tsv(train_file)
            unique_labels = set()
            for (i, line) in enumerate(lines):
                unique_labels.add(line[0])
        elif label_file is None and train_file is None and self.data_dir:
            lines = self._read_tsv(self.data_dir / "train.tsv")
            unique_labels = set()
            for (i, line) in enumerate(lines):
                unique_labels.add(line[0])
        else:
            raise RuntimeError("Cannot find files to generate labels"
                               "You need one of label_file, train_file (full path) or data_dir setup")

        label2idx = {k: v for v, k in enumerate(unique_labels)}
        idx2label = {v: k for k, v in label2idx.items()}

        return unique_labels, label2idx, idx2label

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        raise NotImplementedError(
            "You must use FamilyHistoryRelationDataFormatSep or FamilyHistoryRelationDataFormatOne.")

    @staticmethod
    def _read_tsv(input_file, header=True, quotechar=None):
        """Reads a tab separated value file."""
        # # implemented with csv reader -> deprecate due to artifacts
        # lines = []
        # with open(input_file, "r", encoding="utf-8") as f:
        #     reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        #     for line in reader:
        #         lines.append(line)
        # if header:
        #     lines = lines[1:]
        #
        # return lines

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if header:
                lines = lines[1:]
            return [line.split("\t") for line in lines]


class RelationDataFormatSepProcessor(DataProcessor):
    """
        data format:
            [CLS] sent1 [SEP] sent2 [SEP] : BERT
            <s> sent1 </s> </s> sent2 </s> : RoBERTa, LongFormer
            sent1 <sep> sent2 <sep> <cls>
    """

    def _create_examples_helper(self, lines_idx, set_type, total_special_toks):
        start_idx, lines = lines_idx
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            guid = "{}_{}_{}".format(set_type, start_idx, i)
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            # text after tokenization has a len > max_seq_len:
            # 1. skip all these cases
            # 2. use truncate strategy
            # we adopt truncate way (2) in this implementation as _process_seq_len
            text_a, text_b = self._process_seq_len(text_a, text_b, total_special_toks=total_special_toks)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        if self.tokenizer_type in TOKENIZER_USE_FOUR_SPECIAL_TOKs:
            self.total_special_token_num = 4

        if self.num_core < 2:
            # single process - maybe too slow - replace with multiprocess
            examples = self._create_examples_helper((0, lines), set_type, self.total_special_token_num)
        else:
            # multi-process - assume the first read-in data is the csv title with no data information
            # use multi-cores to process data if you have many long sentences;
            # otherwise single process should be faster
            examples = []
            array_lines = np.array_split(lines, self.num_core)
            with ProcessPoolExecutor(max_workers=self.num_core) as exe:
                for each in exe.map(partial(self._create_examples_helper,
                                            set_type=set_type,
                                            total_special_toks=self.total_special_token_num),
                                    enumerate(array_lines)):
                    examples.extend(each)

        return examples

    @staticmethod
    def _truncate_helper(text):
        tokens = text.split(" ")
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if tk.lower() in SPEC_TAGS]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0:
            return text

        if truncate_space_head > truncate_space_tail:
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)

    def _process_seq_len(self, text_a, text_b, total_special_toks=3):
        """
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        flag = True

        while len(self.tokenizer.tokenize(text_a) + self.tokenizer.tokenize(text_b)) \
                > (self.max_seq_len - total_special_toks):

            if flag:
                text_a = self._truncate_helper(text_a)
            else:
                text_b = self._truncate_helper(text_b)

            flag = not flag

        return text_a, text_b


class RelationDataFormatUniProcessor(DataProcessor):
    """
        data format:
            [CLS] sent1 sent2 [SEP]
    """

    def _create_examples_helper(self, lines_idx, set_type, total_special_toks):
        examples = []
        start_idx, lines = lines_idx
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, start_idx, i)
            text_a = line[1]
            text_a_1 = line[2]
            text_a = " ".join([text_a, text_a_1])
            label = line[0]
            # text after tokenization has a len > max_seq_len:
            # 1. skip all these cases
            # 2. use truncate strategy (truncate from both side) (adopted)
            text_a = self._process_seq_len(text_a)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        if self.tokenizer_type in TOKENIZER_USE_FOUR_SPECIAL_TOKs:
            self.total_special_token_num = 4

        if self.num_core < 2:
            # single process
            examples = self._create_examples_helper((0, lines), set_type, self.total_special_token_num)
        else:
            # multi-process
            examples = []
            array_lines = np.array_split(lines, self.num_core)
            with ProcessPoolExecutor(max_workers=self.num_core) as exe:
                for each in exe.map(partial(self._create_examples_helper,
                                            set_type=set_type,
                                            total_special_toks=self.total_special_token_num),
                                    enumerate(array_lines)):
                    examples.extend(each)

        return examples

    def _process_seq_len(self, text_a):
        """
            see RelationDataFormatSepProcessor._process_seq_len for details
        """
        while len(self.tokenizer.tokenize(text_a)) > (self.max_seq_len - 2):
            w1 = text_a.split(" ")
            t1, t2, t3, t4 = [idx for (idx, w) in enumerate(w1) if w.lower() in SPEC_TAGS]
            ss1, mid1, se1 = 0, (len(w1) - 1) // 2, len(w1) - 1

            a1 = t1 - ss1
            b1 = se1 - t4
            c1 = mid1 - t2
            d1 = t3 - mid1
            m_idx = max(a1, b1, c1, d1)
            if a1 == m_idx:
                w1.pop(0)
            elif b1 == m_idx:
                w1.pop(-1)
            elif c1 == m_idx:
                w1.pop((t2 + c1 // 2))
            else:
                w1.pop((t3 - d1 // 2))

            text_a = " ".join(w1)

        return text_a
