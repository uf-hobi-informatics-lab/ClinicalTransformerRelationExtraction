"""
Post processing

Using this script to merge the system prediction with the entities
The output format will be in BRAT

We will automatically align the predictions to entity pairs and file ids
The results will be write out with the entity information into a new file
We will not copy the original text to the results output dir
"""
import argparse
from pathlib import Path
import numpy as np
from io_utils import load_text, save_text
from collections import defaultdict
from data_format_conf import NON_RELATION_TAG, BRAT_REL_TEMPLATE


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


def map_results(preds, maps):
    llp = len(preds)
    llm = len(maps)
    assert llp == llm, \
        "prediction results and mappings should have same amount data, but got preds: {} and maps: {}".format(llp, llm)
    mapped_preds = defaultdict(list)
    prev_fid = "no previous file id"
    rel_idx = 1
    for each in zip(maps, preds):
        rel_type = each[1]
        # if rel_type == NON_RELATION_TAG:
        if rel_type == "Not-Rel":
            continue
        arg1, arg2, fid = each[0]
        if prev_fid != fid:
            prev_fid = fid
            rel_idx = 1
        brat_res = BRAT_REL_TEMPLATE.format(rel_idx, rel_type, arg1, arg2)
        mapped_preds[fid].append(brat_res)
        rel_idx += 1
    return mapped_preds


def output_results(mapped_predictions, entity_data_dir, output_dir):
    entity_data_dir = Path(entity_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fid in entity_data_dir.glob("*.ann"):
        fid_key = fid.stem
        ofn = output_dir / "{}.ann".format(fid_key)
        entities = load_text(fid).strip()
        if fid_key in mapped_predictions:
            rels = mapped_predictions[fid_key]
            rels = "\n".join(rels)
            outputs = "\n".join([entities, rels])
            save_text(outputs, ofn)
        else:
            save_text(entities, ofn)


def app(args):
    mappings = load_mappings(args.test_data)
    predictions = load_predictions(args.predict_result_file)
    mapped_predictions = map_results(predictions, mappings)
    output_results(mapped_predictions, args.entity_data_dir, args.brat_result_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    # TODO change test_data to mapping file
    parser.add_argument("--test_data", type=str, required=True,
                        help="The test data file in which we need to read the maps")
    parser.add_argument("--entity_data_dir", type=str, required=True,
                        help="The annotation files with all the entities")
    parser.add_argument("--predict_result_file", type=str, required=True,
                        help="prediction results")
    parser.add_argument("--brat_result_output_dir", type=str, required=True,
                        help="prediction results")
    pargs = parser.parse_args()
    app(pargs)
