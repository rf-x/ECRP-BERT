import pickle, json, decimal, math
import numpy as np

def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))


def metrics(truth, prediction):
    assert len(truth) == len(prediction)
    num_predicted = sum(prediction == 1)
    num_ground_truth = sum(truth == 1)
    num_correct = 0
    for tru, pred in zip(truth, prediction):
        num_correct += int(tru) & int(pred)
    p = num_correct / (num_predicted + 1e-8)
    r = num_correct / (num_ground_truth + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    m = (f,p,r)
    return m


def pair_extraction(doc_id_all, doc_pairs_all, clause1_id_all, clause2_id_all, pairs_pred_all):
    doc_id_all, doc_pairs_all, clause1_id_all, clause2_id_all, pairs_pred_all = np.array(doc_id_all), \
                np.array(doc_pairs_all), np.array(clause1_id_all), np.array(clause2_id_all), np.array(pairs_pred_all)
    data = {}
    for doc_id, pair in zip(doc_id_all, doc_pairs_all):
        if doc_id not in data:
            data[doc_id] = {}
            data[doc_id]['pair'] = pair
            data[doc_id]['pred'] = []
    indices = np.where(pairs_pred_all==1)[0]
    for i in indices:
        data[doc_id_all[i]]['pred'].append([int(clause1_id_all[i]), int(clause2_id_all[i])])
    return data


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js

def write_json(data, json_file):
    data = json.dumps(data)
    with open(json_file, 'w', encoding='utf-8') as fw:
        fw.write(data)

