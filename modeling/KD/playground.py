import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseKD4Rec


class RCEKD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rcekd"
        self.tau = args.mkd_tau
        self.K = args.mkd_K
        self.beta = args.mkd_beta
        self.T = args.mkd_T
        self.L = args.mkd_L
        self.mxK = args.mkd_mxK
        self.sample_rank = args.sample_rank
        self.T_topk_dict = self.get_topk_dict(self.teacher, self.K)
        if self.sample_rank:
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)

    def get_topk_dict(self, model, mxK):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = model.get_all_ratings()
            _, topk_dict = torch.topk(inter_mat, mxK, dim=-1)
        return topk_dict.type(torch.LongTensor).cuda()

    # https://discuss.pytorch.org/t/find-indexes-of-elements-from-one-tensor-that-matches-in-another-tensor/147482/3
    def rowwise_index(self, source, target):
        idx = (target.unsqueeze(1) == source.unsqueeze(2)).nonzero()
        idx = idx[:, [0, 2]]
        return idx

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_topk_dict = self.get_topk_dict(self.student, self.mxK)
            self.interesting_items = torch.zeros((self.num_users, self.L)).long()
            if self.sample_rank:
                samples = torch.multinomial(self.ranking_mat, self.L, replacement=False)
            else:
                weight_matrix = torch.zeros((self.num_users, self.mxK)).cuda()
                itemT_rankS = self.rowwise_index(self.T_topk_dict, S_topk_dict)
                weight_matrix[itemT_rankS[:, 0], itemT_rankS[:, 1]] += 1
                weight_matrix = torch.minimum(torch.cumsum(weight_matrix.flip(-1), dim=-1).flip(-1), torch.tensor(50.))
                weight_matrix = torch.exp((weight_matrix + 1) / self.T)
                samples = torch.multinomial(weight_matrix, self.L, replacement=False)
            for user in range(self.num_users):
                self.interesting_items[user] = S_topk_dict[user][samples[user]]
            self.interesting_items = self.interesting_items.cuda()
            self.itemS = S_topk_dict[:, :self.K]

    # https://stackoverflow.com/questions/74946537/can-i-apply-torch-isin-to-each-row-in-2d-tensor-without-loop
    def rowwise_isin(self, tensor_1, target_tensor):
        matches = (tensor_1.unsqueeze(2) == target_tensor.unsqueeze(1))
        result = torch.sum(matches, dim=2, dtype=torch.bool)
        return result

    def get_loss(self, *params):
        batch_users = params[0]
        itemS = self.itemS[batch_users]
        itemT = self.T_topk_dict[batch_users]
        item_interesting = self.interesting_items[batch_users]
        logit_S_itemS = self.student.forward_multi_items(batch_users, itemS) / self.tau
        logit_S_itemT = self.student.forward_multi_items(batch_users, itemT) / self.tau
        logit_S_interesting = self.student.forward_multi_items(batch_users, item_interesting) / self.tau
        logit_T_itemS = self.teacher.forward_multi_items(batch_users, itemS) / self.tau
        logit_T_itemT = self.teacher.forward_multi_items(batch_users, itemT) / self.tau

        exp_logit_T_itemS = torch.exp(logit_T_itemS)
        Z_T = exp_logit_T_itemS.sum(-1, keepdim=True)
        prob_T_itemS = exp_logit_T_itemS / Z_T
        loss_itemS = F.cross_entropy(logit_S_itemS, prob_T_itemS, reduction='none')

        logit_T_interesting = self.teacher.forward_multi_items(batch_users, item_interesting) / self.tau
        exp_logit_T_interesting = torch.exp(logit_T_interesting)
        exp_logit_T_itemT = torch.exp(logit_T_itemT)
        mask = self.rowwise_isin(itemT, item_interesting)
        exp_logit_T_itemT[mask] = 0
        mask2 = self.rowwise_isin(itemT, itemS)
        exp_logit_T_itemT[mask2] = 0
        Z_T = exp_logit_T_interesting.sum(-1, keepdim=True) + exp_logit_T_itemT.sum(-1, keepdim=True)
        prob_T_all = torch.cat([exp_logit_T_interesting, exp_logit_T_itemT], dim=-1) / Z_T
        exp_logit_S_itemT = torch.exp(logit_S_itemT)
        exp_logit_S_itemT = exp_logit_S_itemT * (1. - mask.float()) * (1. - mask2.float())
        exp_logit_S_interesting = torch.exp(logit_S_interesting)
        Z_S = exp_logit_S_interesting.sum(-1, keepdim=True) + exp_logit_S_itemT.sum(-1, keepdim=True)
        logit_S_all = torch.cat([logit_S_interesting, logit_S_itemT], dim=-1)
        loss_itemT = -(prob_T_all * (logit_S_all - torch.log(Z_S))).sum(-1)

        overlap = mask.float().mean(-1)
        weight = torch.exp(-self.beta * overlap)
        loss = ((1 - weight) * loss_itemS + weight * loss_itemT).sum()
        return loss
