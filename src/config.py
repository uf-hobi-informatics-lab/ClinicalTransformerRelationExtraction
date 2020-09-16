from transformers import (BertConfig, RobertaConfig, XLNetConfig, AlbertConfig,
                          BertTokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer)
from models import (BertForRelationIdentification, RoBERTaForRelationIdentification,
                    XLNetForRelationIdentification, AlbertForRelationIdentification)


EN1_START = "[s1]"
EN1_END = "[e1]"
EN2_START = "[s2]"
EN2_END = "[e2]"
SPEC_TAGS = [EN1_START, EN1_END, EN2_START, EN2_END]
MODEL_REQUIRE_SEGMENT_ID = {'bert', 'xlnet', 'albert'}
MODEL_DICT = {
    "bert": (BertForRelationIdentification, BertConfig, BertTokenizer),
    "roberta": (RoBERTaForRelationIdentification, RobertaConfig, RobertaTokenizer),
    "xlnet": (XLNetForRelationIdentification, XLNetConfig, XLNetTokenizer),
    "albert": (AlbertForRelationIdentification, AlbertConfig, AlbertTokenizer)
}
