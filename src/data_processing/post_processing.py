"""
Post processing

Using this script to merge the system prediction with the entities
The output format will be in BRAT

We will automatically align the predictions to entity pairs and file ids
The results will be write out with the entity information into a new file
We will not copy the original text to the results output dir
"""
# import logger from upper level dir
import os
import sys
# TODO: This is so bad. Need a better way to add this path
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from pathlib import Path
import shutil
import argparse
import numpy as np
from utils import TransformerLogger
from data_processing.io_utils import load_text, save_text, pkl_load
from collections import defaultdict
from data_processing.data_format_conf import NON_RELATION_TAG, BRAT_REL_TEMPLATE
import traceback


sys.path.append(Path(os.path.abspath(__file__)).parent.parent.as_posix())


def load_mappings(map_file):
    maps = []
    text = load_text(map_file)
    for idx, line in enumerate(text.strip().split("\n")):
        if idx == 0:
            continue
        info = line.split("\t")
        maps.append(info[-3:])

    return maps


def load_predictions(result_file):
    results = []
    text = load_text(result_file)
    for each in text.strip().split("\n"):
        results.append(each.strip())

    return results


def map_results(res):
    mapped_preds = defaultdict(list)
    prev_fid = "no previous file id"
    rel_idx = 1

    for each in res:
        fid, rt, arg1, arg2 = each
        if prev_fid != fid:
            prev_fid = fid
            rel_idx = 1
        brat_res = BRAT_REL_TEMPLATE.format(rel_idx, rt, arg1, arg2)
        mapped_preds[fid].append(brat_res)
        rel_idx += 1

    return mapped_preds


def output_results(mapped_predictions, entity_data_dir, output_dir, copy_ann=True):
    entity_data_dir = Path(entity_data_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fid in entity_data_dir.glob("*.ann"):
        fid_key = fid.stem
        ofn = output_dir / "{}.ann".format(fid_key)        
        if fid_key in mapped_predictions:
            entities = load_text(fid).strip()
            rels = mapped_predictions[fid_key]
            rels = "\n".join(rels)
            outputs = "\n".join([entities, rels])
            save_text(outputs, ofn)
        elif copy_ann:
            shutil.copy(fid, ofn)


def combine_maps_predictions_mul(args):
    comb_map_pred = []

    for mf, pf in zip(args.test_data_file, args.predict_result_file):
        maps = load_mappings(mf)
        preds = load_predictions(pf)
        llp = len(preds)
        llm = len(maps)
        assert llp == llm, \
            f"prediction results and mappings should have same amount data, but got preds: {llp} and maps: {llm}"
        for m, rel_type in zip(maps, preds):
            if rel_type == args.neg_type:
                continue
            arg1, arg2, fid = m
            comb_map_pred.append((fid, rel_type, arg1, arg2))

    comb_map_pred.sort(key=lambda x: x[0])
    return comb_map_pred


def load_mappings_bin(map_file):
    maps = []
    text = load_text(map_file)
    for idx, line in enumerate(text.strip().split("\n")):
        if idx == 0:
            continue
        info = line.split("\t")
        maps.append(info[-5:])

    return maps


def combine_maps_predictions_bin(args):
    if not args.type_map:
        raise RuntimeError("no type maps (entity-relation) provided. See help.")
    type_maps = pkl_load(args.type_map)

    comb_map_pred = []

    for mf, pf in zip(args.test_data_file, args.predict_result_file):
        maps = load_mappings_bin(mf)
        preds = load_predictions(pf)
        llp = len(preds)
        llm = len(maps)
        assert llp == llm, \
            f"prediction results and mappings should have same amount data, but got preds: {llp} and maps: {llm}"
        for m, rel_type in zip(maps, preds):
            if rel_type == args.neg_type:
                continue
            en_type_1, en_type_2, arg1, arg2, fid = m
            real_rel_type = type_maps[(en_type_1, en_type_2)]
            comb_map_pred.append((fid, real_rel_type, arg1, arg2))

    comb_map_pred.sort(key=lambda x: x[0])
    return comb_map_pred


def app(args):
    lltf = len(args.test_data_file)
    llpf = len(args.predict_result_file)

    if not args.neg_type:
        args.neg_type = NON_RELATION_TAG

    args.logger.info("mode: {}; predict file: {}; output: {}".format(
        args.mode,
        args.predict_result_file,
        args.brat_result_output_dir
    ))

    try:
        assert lltf == llpf
    except AssertionError as ex:
        args.logger.error(
            f"test and prediction file number should be same but get test: {lltf} and preduction {llpf}.")
        raise RuntimeError(
            f"test and prediction file number should be same but get test: {lltf} and preduction {llpf}.")

    if args.mode == "mul":
        combined_results = combine_maps_predictions_mul(args)
    elif args.mode == "bin":
        combined_results = combine_maps_predictions_bin(args)
    else:
        args.logger.error("expect mode to be mul or bin but get {}".format(args.mode))
        raise RuntimeError("expect mode to be mul or bin but get {}".format(args.mode))

    try:
        combined_results = map_results(combined_results)
        output_results(combined_results, args.entity_data_dir, args.brat_result_output_dir, copy_ann=args.copy_ann)
    except Exception as ex:
        traceback.print_exc()
        args.logger.error(traceback.format_exc())

def argparser(args=None):
    parser = argparse.ArgumentParser()
    # parse arguments
    """
        To input multiple test data and prediction files, using following syntax in terminal;
        You need to make sure the files order between test and prediction is correct
        
        bash:
            python post_processing.py --test_data_file tf1.txt --test_data_file tf2.txt --predict_result_file res1.txt
                --predict_result_file res2.txx
        
        in the program:
            args.test_data_file = ['tf1.txt', 'tf2.txt']
            args.predict_result_file = ['res1.txt', 'res2.txt']
            
        if use bin model, you need a map file to map positive relation to its relation type.
        We use entity type pair as key to conduct this mapping
        example:
            (ADE, Drug): Drug-ADE
    """
    parser.add_argument("--mode", type=str, default='mul', required=True,
                        help="we have two mode for binary (bin) and multiple (mul) classes classification")
    parser.add_argument("--neg_type", type=str, default=None,
                        help="the type used for representing non-relation.")
    parser.add_argument("--type_map", type=str, default=None,
                        help="a map of entity pair types to relation types (only use when mode is bin)")
    parser.add_argument("--test_data_file", type=str, nargs='+', required=True,
                        help="The test data file in which we need to read the maps; available to accept multiple files")
    parser.add_argument("--entity_data_dir", type=str, required=True,
                        help="The annotation/NER output files with only the entities. Used for output NER and RE.")
    parser.add_argument("--predict_result_file", nargs='+', type=str, required=True,
                        help="prediction results; available to accept multiple files")
    parser.add_argument("--brat_result_output_dir", type=str, required=True,
                        help="prediction results")
    parser.add_argument("--log_file", default="./log.txt", type=str,
                        help="where to save the log information")
    parser.add_argument("--copy_ann", default=True, type=bool,
                        help="whether to copy orignial ann if no relation exists")

    if args is None:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)
        
    parsed_args.logger = TransformerLogger(logger_file=parsed_args.log_file, logger_level='i').get_logger()

    return parsed_args

if __name__ == '__main__':
    pargs = argparser()
    app(pargs)
