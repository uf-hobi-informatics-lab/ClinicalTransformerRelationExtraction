"""
Post processing

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
