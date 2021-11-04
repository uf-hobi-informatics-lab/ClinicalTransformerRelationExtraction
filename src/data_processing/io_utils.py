import pickle as pkl
import json


def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt


def save_text(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)


def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)
