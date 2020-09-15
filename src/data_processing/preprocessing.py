"""
The preprocessing module is used to convert annotation (brat formatted) to training and test data
We also provide a implementation for generating datasets for 5-CV (train.tsv and dev.tsv)
Requirement:
1. raw text files
2. Annotations of Entities for candidate relation generation (we present a relation as a pair of two entities)
"""
import argparse
from pathlib import Path
import numpy as np


def app(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory. Should have train.tsv")
    pargs = parser.parse_args()
    app(pargs)

