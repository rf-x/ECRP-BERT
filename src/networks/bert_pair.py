import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from networks.focal_loss import FocalLoss

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.feet_dim = configs.feat_dim
        self.is_pos = configs.is_pos
        self.is_focal_loss = configs.is_focal_loss
        self.pos_emb_dim = configs.pos_emb_dim
        self.distance = configs.distance
        self.dp = configs.dp
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.pos_layer = nn.Embedding(2 * self.distance + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)
        self.pred_e = nn.Linear(self.feet_dim, 1)
        self.pred_c = nn.Linear(self.feet_dim, 1)
        self.pred_p = nn.Sequential(
            nn.Linear(self.feet_dim + self.pos_emb_dim, self.feet_dim),
            nn.ReLU(),
            nn.Dropout(self.dp),
            nn.Linear(self.feet_dim, 1)
        )
        self.focal_loss = FocalLoss()

    def forward(self, input_token_b, segments_b, mask_b, clause1_len_b, clause1_id_b, clause2_id_b):
        output = self.bert(input_ids=input_token_b.to(DEVICE),
                           attention_mask=mask_b.to(DEVICE),
                           token_type_ids= segments_b.to(DEVICE))
        # tuple ((B, L, H), (B, H))
        clause1_len_b = clause1_len_b.to(DEVICE)
        clause1_id_b = clause1_id_b.to(DEVICE)
        clause2_id_b = clause2_id_b.to(DEVICE)

        rel_pos = clause1_id_b - clause2_id_b + self.distance
        rel_pos = self.pos_layer(rel_pos)

        if self.is_pos:
            pair_h = torch.cat([output[-1], rel_pos], dim=-1)
        else:
            pair_h = output[-1]

        emotion_h = output[0][:,1]  #(B,H)
        cause_h = self.cause_select(output, clause1_len_b)

        emotion_pred = self.pred_e(emotion_h)
        cause_pred = self.pred_c(cause_h)
        pair_pred = self.pred_p(pair_h)
        return emotion_pred, cause_pred, pair_pred

    def cause_select(self, output, clause1_len_b):
        hidden_state = output[0]    # (B, L, H)
        index = clause1_len_b.unsqueeze(1) + 1  # (B, 1)
        dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), hidden_state.size(2)) #(B,1,H)
        cause_h = hidden_state.gather(1, dummy)
        return cause_h.squeeze(1)

    # def loss(self, pred_emotion, pred_cause, pred_pair, true_emotion, true_cause, true_pair, clause1_id, clause2_id):
    #     pred_emotion = pred_emotion.squeeze(1) # (B)  (B)
    #     pred_cause = pred_cause.squeeze(1)
    #     pred_pair = pred_pair.squeeze(1)
    #
    #     # loss
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     loss_e = criterion(pred_emotion, true_emotion.to(DEVICE))
    #     loss_c = criterion(pred_cause, true_cause.to(DEVICE))
    #     loss_p = criterion(pred_pair, true_pair.to(DEVICE))
    #
    #     # pred
    #     ones = torch.ones_like(pred_emotion)
    #     zeros = torch.zeros_like(pred_emotion)
    #
    #     pred_emotion = F.sigmoid(pred_emotion)
    #     pred_cause = F.sigmoid(pred_cause)
   

    def loss(self, pred_emotion, pred_cause, pred_pair, true_emotion, true_cause, true_pair, clause1_id, clause2_id):
        pred_emotion = pred_emotion.squeeze(1)  # (B)  (B)
        pred_cause = pred_cause.squeeze(1)
        pred_pair = pred_pair.squeeze(1)

        # focal loss
        if self.is_focal_loss:
            loss_e = self.focal_loss(pred_emotion, true_emotion.to(DEVICE))
            loss_c = self.focal_loss(pred_cause, true_cause.to(DEVICE))
            loss_p = self.focal_loss(pred_pair, true_pair.to(DEVICE))

        # loss
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            loss_e = criterion(pred_emotion, true_emotion.to(DEVICE))
            loss_c = criterion(pred_cause, true_cause.to(DEVICE))
            loss_p = criterion(pred_pair, true_pair.to(DEVICE))

        # pred
        ones = torch.ones_like(pred_emotion)
        zeros = torch.zeros_like(pred_emotion)

        pred_emotion = F.sigmoid(pred_emotion)
        pred_cause = F.sigmoid(pred_cause)
        pred_pair = F.sigmoid(pred_pair)

        pred_e = torch.where(pred_emotion > 0.5, ones, zeros)
        pred_c = torch.where(pred_cause > 0.5, ones, zeros)
        pred_p = torch.where(pred_pair > 0.5, ones, zeros)

        return loss_e, loss_c, loss_p, pred_e, pred_c, pred_p