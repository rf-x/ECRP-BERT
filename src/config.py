import torch
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = './data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'


class Config(object):
    def __init__(self):
        self.n_folds = 10
        self.is_pos = True       # position embedding
        self.is_multi_task = True # multi-task learning
        self.is_focal_loss = False # focal loss

        self.bert_cache_path = 'bert-base-chinese'
        self.checkpoint = 'checkpoint'
        if self.is_focal_loss:
            self.checkpoint = 'checkpoint' + '_focal'

        self.distance = 4  # the relative distance between clauses
        self.data_path_prefix = 'pair_data_4'
        if self.is_pos is not True:
            self.data_path = self.data_path_prefix + '_no_pos'   # result data path
        elif self.is_multi_task is not True:
            self.data_path = self.data_path_prefix + '_no_multi'
        else:
            self.data_path = self.data_path_prefix

        self.feat_dim = 768
        self.pos_emb_dim = 50 if self.is_pos else 0

        self.epochs = 4
        self.lr = 1e-5
        self.batch_size = 8
        self.test_batch_size= 32
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8


