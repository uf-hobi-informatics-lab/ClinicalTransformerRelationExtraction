"""
The preprocessing module is used to convert annotation (brat formatted) to training and test data
We also provide a implementation for generating datasets for 5-CV (train.tsv and dev.tsv)
Requirement:
1. raw text files
2. Annotations of Entities for candidate relation generation (we present a relation as a pair of two entities)
3. We only support brat format.
if your data is in different annotation format, you can convert the annotation into brat first
"""
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os


def app(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="The data directory for annotations (entity and relation for training; entity for test)")
    parser.add_argument("--raw_text_dir", type=str, required=True,
                        help="")
    pargs = parser.parse_args()
    app(pargs)

