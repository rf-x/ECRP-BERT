from utils.utils import *
from config import *
import os

def eval_score(num_correct, num_predicted, num_ground_truth):
    p = num_correct / (num_predicted + 1e-8)
    r = num_correct / (num_ground_truth + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    return [f, p, r]


if __name__ == '__main__':
    configs = Config()
    n_folds = 10
    metric_folds = {'p': [], 'e': [], 'c': []}
    for fold_id in range(1, n_folds+1):
        read_path = os.path.join(configs.checkpoint, configs.data_path, 'fold{}_result.txt'.format(fold_id))
        print('===== Eval: ' + read_path + '=======')
        data = read_json(read_path)
        num_predicted, num_ground_truth, num_correct = {}, {}, {}
        num_predicted['p'], num_ground_truth['p'], num_correct['p'] = 0, 0, 0
        num_predicted['e'], num_ground_truth['e'], num_correct['e'] = 0, 0, 0
        num_predicted['c'], num_ground_truth['c'], num_correct['c'] = 0, 0, 0
        for _, value in data.items():   # each doc
            pair = value['pair']
            pred = value['pred']
            emotion_pair, emotion_pred = set(), set()
            cause_pair, cause_pred = set(), set()

            # the ground truth of emotion and cause
            for p_pair in pair:
                emotion_pair.add(p_pair[0])
                cause_pair.add(p_pair[1])

            for p_pred in pred:
                # pair
                if p_pred in pair:
                    num_correct['p'] += 1
                # the prediction of emotion and cause
                emotion_pred.add(p_pred[0])
                cause_pred.add(p_pred[1])

            num_ground_truth['e'] += len(emotion_pair)
            num_ground_truth['c'] += len(cause_pair)
            num_ground_truth['p'] += len(pair)
            num_predicted['e'] += len(emotion_pred)
            num_predicted['c'] += len(cause_pred)
            num_predicted['p'] += len(pred)

            num_correct['e'] += len(emotion_pair & emotion_pred)
            num_correct['c'] += len(cause_pair & cause_pred)

        metric_p = eval_score(num_correct['p'], num_predicted['p'], num_ground_truth['p'])
        metric_e = eval_score(num_correct['e'], num_predicted['e'], num_ground_truth['e'])
        metric_c = eval_score(num_correct['c'], num_predicted['c'], num_ground_truth['c'])

        metric_folds['p'].append(metric_p)
        metric_folds['e'].append(metric_e)
        metric_folds['c'].append(metric_c)

    p_avg = np.mean(np.array(metric_folds['p']), axis=0)
    e_avg = np.mean(np.array(metric_folds['e']), axis=0)
    c_avg = np.mean(np.array(metric_folds['c']), axis=0)

    print(p_avg)
    print(e_avg)
    print(c_avg)
