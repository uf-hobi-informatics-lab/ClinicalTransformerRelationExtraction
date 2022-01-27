import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import traceback
from collections import defaultdict


def try_catch_annotator(func):
    def try_catch(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            traceback.print_exc()
            return None
    return try_catch


class TransformerLogger:
    LOG_LVLs = {
        'i': logging.INFO,
        'd': logging.DEBUG,
        'e': logging.ERROR,
        'w': logging.WARN
    }

    def __init__(self, logger_file=None, logger_level='d'):
        self.lf = logger_file
        self.lvl = logger_level

    def set_log_info(self, logger_file, logger_level):
        self.lf = logger_file
        self.lvl = logger_level

    def _create_logger(self, logger_name=""):
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        logger.setLevel(self.LOG_LVLs[self.lvl])
        if self.lf:
            fh = logging.FileHandler(self.lf)
            fh.setFormatter(formatter)
            fh.setLevel(self.LOG_LVLs[self.lvl])
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(self.LOG_LVLs[self.lvl])
            logger.addHandler(ch)

        return logger

    def get_logger(self):
        return self._create_logger("Transformer_Relation_Extraction")


class PRF:
    def __init__(self):
        self.tp = 0
        self.fp = 0

    def __repr__(self):
        return f"tp: {self.tp}; fp: {self.fp}"


def calc(tp, tp_fp, tp_tn):
    if tp_fp != 0:
        pre = tp / tp_fp
    else:
        pre = 0

    if tp_tn == 0:
        rec = 0
    else:
        rec = tp / tp_tn

    if pre == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * pre * rec / (pre + rec)

    return round(pre, 4), round(rec, 4), round(f1, 4)


def measure_prf(preds, gs_labels, non_rel_label):
    res = dict()
    temp = defaultdict(PRF)
    total_tp, total_tp_fp, total_tp_tn = 0, 0, 0
    tn_dict = defaultdict(lambda: 0)

    assert preds == gs_labels, \
        f"prediction and gold standard is not equal, prediction: {len(preds)}; gs: {len(gs_labels)}"

    labels = set(gs_labels)
    for l in labels:
        for p, g in zip(preds, gs_labels):
            if g == l:
                tn_dict[l] += 1
            if g == p == l:
                temp[l].tp += 1
            elif g != l and p == l:
                temp[l].fp += 1

    for l in labels:
        if l == non_rel_label:
            continue
        tp, fp = temp[l].tp, temp[l].fp
        tp_fp = tp + fp
        tp_tn = tn_dict[l]
        res[l] = calc(tp, tp_fp, tp_tn)

        total_tp += tp
        total_tp_fp += tp_fp
        total_tp_tn += tp_tn

    res['micro_average_pre_rec_f1'] = calc(total_tp, total_tp_fp, total_tp_tn)
    f1 = res['micro_average_pre_rec_f1'][-1]

    return res, f1


def acc_and_f1(labels, preds, label2idx, non_rel_label):
    acc = accuracy_score(labels, preds)

    includes = [i for l, i in label2idx.items() if l != non_rel_label]

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=includes, average="micro")

    return acc, f"precision: {p}; recall: {r}", f1
