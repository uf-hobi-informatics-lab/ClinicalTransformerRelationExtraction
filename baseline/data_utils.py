import pickle as pkl
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize


def pkl_save(data, fn):
    with open(fn, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def pkl_load(fn):
    with open(fn, "r") as f:
        data = pkl.load(f)
    return data


def read_brat(fn):
    pass


def nltk_tokenization_engine(text):
    """
     input text
     output tokens: [token_text, (char offset start, char offset end)]
    """
    nsents = []
    tokenizer = WhitespaceTokenizer()
    prev_span = 0
    for sent in sent_tokenize(text):
        tokens = tokenizer.tokenize(sent)
        spans = [(span[0]+prev_span, span[1]+prev_span) for span in tokenizer.span_tokenize(sent)]
        prev_span = spans[-1][-1] + 1
        nsents.append(list(zip(tokens, spans)))

    return nsents
