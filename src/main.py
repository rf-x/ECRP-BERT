import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import collections
import torch
from config import *
from data_loader import *
from networks.bert_pair import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *


def main(configs, fold_id):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = Network(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.pred_e.parameters()) + list(model.pred_c.parameters()) + list(model.pred_p.parameters()) +\
                  list(model.pos_layer.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},  # not bias or 'bias', 'LayerNorm.weight'
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]
    optimizer = AdamW(params, lr=configs.lr)

    num_steps_all = len(train_loader) * configs.epochs

    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)


    best_metric_ec = (-1, -1, -1)
    best_model = None
    best_doc_data = None
    model.zero_grad()
    for epoch in range(1, configs.epochs+1):
        print('===== epoch {} ====='.format(epoch))
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_pairs_b, doc_id_b, clause1_id_b, clause2_id_b, y_emotions_b, y_causes_b, \
            y_pair_b, input_token_b, input_token_len_b, segments_idx_b, input_mask_b, clause1_token_len_b = batch
            pred_emotion, pred_cause, pred_pair = model(input_token_b, segments_idx_b, input_mask_b,
                                                        clause1_token_len_b, clause1_id_b, clause2_id_b)

            loss_e, loss_c, loss_p, _, _, _ = model.loss(pred_emotion, pred_cause, pred_pair,
                                                         y_emotions_b, y_causes_b, y_pair_b, clause1_id_b, clause2_id_b)

            loss = loss_e + loss_c + loss_p if configs.is_multi_task else loss_p
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if train_step % 100== 0:
                print("{}/{} loss: {}".format(train_step, len(train_loader) , loss.item()))

            if train_step % 300== 0 or train_step == len(train_loader):
                print('===== Test =====')
                with torch.no_grad():
                    model.eval()
                    metric_ec, doc_data = inference_one_epoch(configs, test_loader, model, best_metric_ec)
                print('Pair: F: {:.4f}, P: {:.4f}, R: {:.4f}'.format(metric_ec[0], metric_ec[1], metric_ec[2]))
                if metric_ec[0] > best_metric_ec[0]:
                    best_metric_ec = metric_ec
                    best_model = model
                    best_doc_data = doc_data
                print('===== Train =====')

        print('\nepoch: {} best_metric ||  F: {:.4f}, P: {:.4f}, R: {:.4f}\n'.format(epoch, best_metric_ec[0], best_metric_ec[1],
                                                                         best_metric_ec[2]))
    return best_metric_ec, best_model, best_doc_data


def inference_one_epoch(configs, batches, model, best_metric_ec):
    doc_id_all, clause1_id_all, clause2_id_all = [], [], []
    doc_pairs_all, pairs_pred_all, pairs_y_all = [], [], []
    doc_data = None
    for batch in batches:
        pred_p, y_p, doc_id, doc_pairs, clause1_id, clause2_id = inference_one_batch(configs, batch, model)
        pairs_pred_all.extend(pred_p)
        pairs_y_all.extend(y_p)
        doc_pairs_all.extend(doc_pairs)
        doc_id_all.extend(doc_id)
        clause1_id_all.extend(clause1_id)
        clause2_id_all.extend(clause2_id)
    metric_ec = metrics(np.array(pairs_y_all), np.array(pairs_pred_all))
    if metric_ec[0] > best_metric_ec[0]:
        doc_data = pair_extraction(doc_id_all, doc_pairs_all, clause1_id_all, clause2_id_all, pairs_pred_all)
    return metric_ec, doc_data

def inference_one_batch(configs, batch, model):
    doc_pairs_b, doc_id_b, clause1_id_b, clause2_id_b, y_emotions_b, y_causes_b, \
    y_pair_b, input_token_b, input_token_len_b, segments_idx_b, input_mask_b, clause1_token_len_b = batch

    pred_emotion, pred_cause, pred_pair = model(input_token_b, segments_idx_b, input_mask_b, \
                                                clause1_token_len_b, clause1_id_b, clause2_id_b)
    loss_e, loss_c, loss_p, pred_e, pred_c, pred_p = model.loss(pred_emotion, pred_cause, pred_pair, y_emotions_b,
                                                                y_causes_b, y_pair_b, clause1_id_b, clause2_id_b)

    return to_np(pred_p), to_np(y_pair_b), doc_id_b, doc_pairs_b, to_np(clause1_id_b), to_np(clause2_id_b)


if __name__ == '__main__':
    configs = Config()
    n_folds = 2
    metric_folds = []
    for fold_id in range(1, n_folds+1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, model, doc_data = main(configs, fold_id)
        print('F_p: {}. P_p: {}, R_p: {}'.format(float_n(metric_ec[0]), float_n(metric_ec[1]), float_n(metric_ec[2])))

        result_file = os.path.join(configs.checkpoint, configs.data_path, 'fold{}_result.txt'.format(fold_id))
        if not os.path.exists(os.path.join(configs.checkpoint, configs.data_path)):
            os.makedirs(os.path.join(configs.checkpoint, configs.data_path))
        write_json(doc_data, result_file)
        metric_folds.append(metric_ec)
        print('flag: {}'.format(metric_folds))
    metric_ec = np.mean(np.array(metric_folds), axis=0).tolist()
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[0]), float_n(metric_ec[1]), float_n(metric_ec[2])))
    write_json({'ecp': metric_folds, 'average_ecp': metric_ec}, os.path.join(configs.checkpoint, configs.data_path, 'metrics.json'.format(time.time())))