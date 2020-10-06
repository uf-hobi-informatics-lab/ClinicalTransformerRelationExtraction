"""
The preprocessing module is used to convert annotation (brat formatted) to training and test data
We also provide a implementation for generating datasets for 5-CV (train.tsv and dev.tsv)
Requirement:
1. raw text files
2. Annotations of Entities for candidate relation generation (we present a relation as a pair of two entities)
3. We use brat format.
4. we used a self-developed sentence boundary tokenization tool to perform the sentence boundary detection and tokenization
5. we also have a self-developed tool to convert brat to BIO

The key information we need is "nsents" which format is
[[['Admission', (0, 9), (0, 9), (0, 0), 'O'], [next word]], [next sentence]]
for ['Admission', (0, 9), (0, 9), (0, 0), 'O']:
first (0, 9) is the original offset
second (0, 9) is the offset after tokenization
(0, 0) is (sent_idx, token_idx)
sent_idx is index for which sentence the token located
token_idx is index for the token in the current sentence

We have formatted data examples in the readme, you can also skip this script
use your own script to generate the data according to the format in the example
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os
from io_utils import pkl_save, load_text
from data_format_conf import DEID_PATTERN


def app(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="The data directory for annotations (entity and relation for training; entity for test)")
    parser.add_argument("--raw_text_dir", type=str, required=True,
                        help="The clinical notes text directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The formatted data output directory; will generate train.tsv and test.tsv and 5-CV data")
    parser.add_argument("--5cv", action='store_true',
                        help="Whether to generate data for 5-fold cross validation (hyperparameter tunining)")
    parser.add_argument("--mode", type=str, default='mul',
                        help="Whether approach RE as binary or multi-class classification task (bin or mul)")

    pargs = parser.parse_args()
    app(pargs)

