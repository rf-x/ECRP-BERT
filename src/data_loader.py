import sys

sys.path.append('..')
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils.utils import *
from tqdm import tqdm

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True

def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=batch_preprocessing)
    # merges a list of samples to form a mini-batch of Tensor(s)
    return train_loader

def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.test_batch_size,
                                              shuffle=False, collate_fn=batch_preprocessing)
    return data_loader

class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_path = configs.data_path_prefix
        self.data_type = data_type

        self.train_file = join(data_dir, self.data_path, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.data_path, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.data_path, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, self.y_pair_list, self.doc_len_list, self.doc_id_list, \
        self.input_token_list, self.segments_idx_list, self.clause1_token_len_list, self.clause2_token_len_list, \
        self.clauses_token_len_list, self.input_token_len_list, self.clause1_id_list, self.clause2_id_list  \
            = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples = self.doc_couples_list[idx]
        y_emotions = self.y_emotions_list[idx]
        y_causes = self.y_causes_list[idx]
        y_pair = self.y_pair_list[idx]
        doc_len = self.doc_len_list[idx]
        doc_id = self.doc_id_list[idx]
        input_token =self.input_token_list[idx]
        segments_idx = self.segments_idx_list[idx]
        clause1_token_len = self.clause1_token_len_list[idx]
        clause2_token_len = self.clause2_token_len_list[idx]
        clauses_token_len = self.clauses_token_len_list[idx]
        input_token_len = self.input_token_len_list[idx]
        clause1_id = self.clause1_id_list[idx]
        clause2_id = self.clause2_id_list[idx]

        assert input_token_len == len(input_token)
        if len(input_token) > 512:
            input_token, segments_idx, clauses_token_len, input_token_len = \
                self.token_trunk(input_token, segments_idx,clause1_token_len, clause2_token_len,
                            clauses_token_len, input_token_len, doc_couples, doc_len)

        input_token  = torch.LongTensor(input_token)
        segments_idx = torch.LongTensor(segments_idx)
        clause1_token_len = torch.LongTensor([clause1_token_len])
        clause2_token_len = torch.LongTensor([clause2_token_len])
        clauses_token_len = torch.LongTensor([clauses_token_len])

        return doc_couples, y_emotions, y_causes, y_pair, doc_len, doc_id, input_token, segments_idx, \
               clause1_token_len, clause2_token_len, clauses_token_len, input_token_len, clause1_id, clause2_id

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_couples_list = []
        y_emotions_list = []
        y_causes_list = []
        y_pair_list = []
        doc_len_list = []
        doc_id_list = []
        input_token_list = []
        segments_idx_list = []
        clause1_token_len_list = []
        clause2_token_len_list = []
        clauses_token_len_list = []
        input_token_len_list = []
        clause1_id_list = []
        clause2_id_list = []


        data_list =read_json(data_file)

        for doc in tqdm(data_list, desc='Loading %s data' % data_type):
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)

            clause1 = doc['clause_1']
            clause2 = doc['clause_2']
            clauses = doc['clauses']
            clause1_id = int(clause1['clause_id'])
            clause2_id = int(clause2['clause_id'])
            clause1_id_list.append(clause1_id)
            clause2_id_list.append(clause2_id)

            input_str = '[CLS]' + '[CLS]' + clause1['clause'] + '[SEP]' + '[CLS]' + clause2['clause'] + '[SEP]'

            emotion_label = int(int(clause1['clause_id']) in doc_emotions)
            cause_label = int(int(clause2['clause_id']) in doc_causes)
            pair_label = int([int(clause1['clause_id']), int(clause2['clause_id'])] in doc_couples)
            y_emotions_list.append(emotion_label)
            y_causes_list.append(cause_label)
            y_pair_list.append(pair_label)

            for i in range(doc_len):
                clause = clauses[i]
                assert int(clause['clause_id']) == i + 1
                input_str += clause['clause']
            # input [CLS] [CLS] clause1 [SEP] [CLS] clause2 [SEP] clauses
            input_str += '[SEP]'

            index_tokens = self.bert_tokenizer.encode(input_str.strip(), add_special_tokens=False) # str to id

            cls_indices = np.array([i for i, x in enumerate(index_tokens) if x == 101])[1:] # 101 is [CLS]
            sep_indices = np.array([i for i, x in enumerate(index_tokens) if x == 102])  # 102 is [SEP]
            clause1_token_len = sep_indices[0] - cls_indices[0] + 1
            clause2_token_len = sep_indices[1] - cls_indices[1] + 1
            clauses_token_len = sep_indices[-1] - sep_indices[1]
            input_token_len = clause1_token_len + clause2_token_len + clauses_token_len + 1
            input_token_len_list.append(input_token_len)

            # segments_idx = [0] * (1 + clause1_token_len + clause2_token_len)
            # segments_idx.extend([1] * clauses_token_len)
            segments_idx = [0] * (1 + clause1_token_len)
            segments_idx.extend([1]*(clause2_token_len))
            segments_idx.extend([0] * clauses_token_len)

            assert len(segments_idx) == len(index_tokens)
            input_token_list.append(index_tokens)
            clause1_token_len_list.append(clause1_token_len)
            clause2_token_len_list.append(clause2_token_len)
            clauses_token_len_list.append(clauses_token_len)
            segments_idx_list.append(segments_idx)

        return doc_couples_list, y_emotions_list, y_causes_list, y_pair_list, doc_len_list, doc_id_list, \
               input_token_list, segments_idx_list, clause1_token_len_list, clause2_token_len_list, \
               clauses_token_len_list, input_token_len_list, clause1_id_list, clause2_id_list

    def token_trunk(self, input_token, segments_idx, clause1_token_len, clause2_token_len,
                    clauses_token_len, input_token_len, doc_couples, doc_len):
        emotion, cause = doc_couples[0]
        trunk_len = input_token_len - 512
        clauses_token_len -= trunk_len
        if emotion > doc_len / 2 and cause > doc_len / 2:   # truncate from the head
            input_a_length = 1 + clause1_token_len + clause2_token_len
            input_token = input_token[:input_a_length] + input_token[input_a_length+trunk_len:]
            segments_idx = segments_idx[:input_a_length] + segments_idx[input_a_length+trunk_len:]
        else:
            sep_token = input_token[-1]
            input_token = input_token[:input_token_len-trunk_len-1]
            input_token.append(sep_token)
            segments_idx = segments_idx[:input_token_len-trunk_len]
        input_token_len -= trunk_len
        return input_token, segments_idx, clauses_token_len, input_token_len

def batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, y_pair_b, doc_len_b, doc_id_b, input_token_b, segments_idx_b, \
    clause1_token_len_b, clause2_token_len_b, clauses_token_len_b, input_token_len_b, clause1_id_b, clause2_id_b = zip(*batch)

    # input
    input_token_b = pad_sequence(input_token_b, batch_first=True, padding_value=0)
    segments_idx_b = pad_sequence(segments_idx_b, batch_first=True, padding_value=0)

    batch_size, max_len = input_token_b.size()
    input_mask_b = np.zeros([batch_size, max_len], dtype=np.float)
    for i, token_len in enumerate(input_token_len_b):
        input_mask_b[i][:token_len] = 1
    input_mask_b = torch.FloatTensor(input_mask_b)
    y_emotions_b = torch.FloatTensor(y_emotions_b)
    y_causes_b = torch.FloatTensor(y_causes_b)
    y_pair_b = torch.FloatTensor(y_pair_b)
    clause1_token_len_b = torch.LongTensor(clause1_token_len_b)
    clause1_id_b = torch.LongTensor(clause1_id_b)
    clause2_id_b = torch.LongTensor(clause2_id_b)

    return doc_couples_b, doc_id_b, clause1_id_b, clause2_id_b, y_emotions_b, y_causes_b, \
           y_pair_b, input_token_b, input_token_len_b, segments_idx_b, input_mask_b, clause1_token_len_b




