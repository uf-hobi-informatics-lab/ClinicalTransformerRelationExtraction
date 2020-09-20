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


def combine_maps_preditions(args):
    comb_map_pred = []

    for mf, pf in zip(args.test_data_file, args.predict_result_file):
        maps = load_mappings(mf)
        preds = load_predictions(pf)
        llp = len(preds)
        llm = len(maps)
        assert llp == llm, \
            f"prediction results and mappings should have same amount data, but got preds: {llp} and maps: {llm}"
        for m, rel_type in zip(maps, preds):
            if rel_type == NON_RELATION_TAG:
                continue
            arg1, arg2, fid = m
            comb_map_pred.append((fid, rel_type, arg1, arg2))

    # comb_map_pred = sorted(comb_map_pred, key=lambda x: x[0])
    comb_map_pred.sort(key=lambda x: x[0])
    return comb_map_pred


def app(args):
    lltf = len(args.test_data_file)
    llpf = len(args.predict_result_file)
    assert lltf == llpf, \
        f"test and prediction file number should be same but get test: {lltf} and preduction {llpf}."

    combined_results = combine_maps_preditions(args)

    combined_results = map_results(combined_results)

    output_results(combined_results, args.entity_data_dir, args.brat_result_output_dir)


if __name__ == '__main__':
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
    """
    parser.add_argument("--test_data_file", type=str, action='append', required=True,
                        help="The test data file in which we need to read the maps; available to accept multiple files")
    parser.add_argument("--entity_data_dir", type=str, required=True,
                        help="The annotation files with all the entities")
    parser.add_argument("--predict_result_file", action='append', type=str, required=True,
                        help="prediction results; available to accept multiple files")
    parser.add_argument("--brat_result_output_dir", type=str, required=True,
                        help="prediction results")
    pargs = parser.parse_args()

    app(pargs)
