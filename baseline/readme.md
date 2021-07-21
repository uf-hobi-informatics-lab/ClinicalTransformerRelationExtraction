# Relation Extraction Baseline - SVM

## Aim

- implement machine learning based RE pipeline as baseline for comparison
- examine SVM algorithms
- pipeline to generated features based on BRAT annotations
- quick and out-of-box ready
- only for comparison not for production

## Features
- extract general lexical and semantic features like pos-tag and ngrams


## tokenizer
- we use nltk's WhitespaceTokenizer, sent_tokenize for tokenization here
- we use our own tokenizer (customized for clinical text) in experiments when we report results in publication
- check data_utils.nltk_tokenization_engine function for how to implement your own tokenization function to replace nltk