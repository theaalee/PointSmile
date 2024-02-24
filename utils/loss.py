import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import argparse
import numpy as np
import random
import math
from lightly.loss.ntx_ent_loss import NTXentLoss


FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--seed', default=None, type=int, help='random seed')
args = FLAGS.parse_args()

# fix random seeds
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print('Random seed will be fixed to %d' % args.seed)


class contrastive_loss(nn.Module):
    def __init__(self, batch_size, temperature=0.2, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        is_cuda = representations.is_cuda
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        # if self.verbose:
        #     print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            # if self.verbose:
            #     print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * int(representations.size()[0]/2),)).scatter_(0, torch.tensor([i]), 0.0)
            one_for_not_i = one_for_not_i.cuda() if is_cuda else one_for_not_i
            # if self.verbose:
            #     print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            # if self.verbose:
            #     print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            # if self.verbose:
            #     print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = int(representations.size()[0]/2)
        loss = 0.0
        for k in range(0, N):
            # if N <= representations.size():
            loss += l_ij(k, k + N) + l_ij(k + N, k)
            # else
        return 1.0 / (2 * N) * loss


def ChamferDistance(x, y):  # for example, x = batch,2025,3 y = batch,2048,3
    #   compute chamfer distance between tow point clouds x and y
    x_size = x.size()
    y_size = y.size()
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
    y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

    x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
    y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3

    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
    x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
    x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1

    x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
    x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
    x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = torch.mean(chamfer_distance)
    return chamfer_distance


class ChamferLoss(nn.Module):
    # chamfer distance loss
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        return ChamferDistance(x, y)


# class ClusterLoss_1(nn.Module):
#     def __init__(self, class_num, temperature: float = 0.5):
#         super(ClusterLoss_1, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#         self.similarity_f_1 = nn.CosineSimilarity(dim=1)
#         self.class_num = class_num
#
#     def forward(self, out0, out1):
#         device = out0.device
#
#         p_i = out0.sum(0).view(-1)
#         p_i /= p_i.sum()
#         ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
#         p_j = out1.sum(0).view(-1)
#         p_j /= p_j.sum()
#         ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
#         ne_loss = ne_i + ne_j
#
#         out0 = out0.t()
#         out1 = out1.t()
#
#         c_00 = torch.cat((out0, out0), dim=0)
#         c_01 = torch.cat((out0, out1), dim=0)
#         c_10 = torch.cat((out1, out0), dim=0)
#         c_11 = torch.cat((out1, out1), dim=0)
#         logits_00 = self.similarity_f(out0.unsqueeze(1), out0.unsqueeze(0)) / self.temperature  # 16
#         logits_01 = self.similarity_f(out0.unsqueeze(1), out1.unsqueeze(0)) / self.temperature
#         logits_10 = self.similarity_f(out1.unsqueeze(1), out0.unsqueeze(0)) / self.temperature
#         logits_11 = self.similarity_f(out1.unsqueeze(1), out1.unsqueeze(0)) / self.temperature
#         # logits_00 = torch.einsum('nc,mc->nm', out0, out0) / self.temperature
#         # logits_01 = torch.einsum('nc,mc->nm', out0, out1) / self.temperature
#         # logits_10 = torch.einsum('nc,mc->nm', out1, out0) / self.temperature
#         # logits_11 = torch.einsum('nc,mc->nm', out1, out1) / self.temperature
#
#         # initialize labels and masks
#         labels = torch.arange(self.class_num, device=device, dtype=torch.long)
#         masks = torch.ones_like(logits_00).bool()
#         masks.scatter_(dim=1, index=labels.unsqueeze(1), value=False)
#         # remove similarities of samples to themselves
#         logits_00 = logits_00[masks].view(self.class_num, -1)
#         logits_11 = logits_11[masks].view(self.class_num, -1)
#         # concatenate logits
#         # the logits tensor in the end has shape (2*n, 2*m-1)
#         logits_0100 = torch.cat([logits_01, logits_00], dim=1)
#         logits_1011 = torch.cat([logits_10, logits_11], dim=1)
#
#         # positive_clusters, negative_clusters
#         logits = torch.cat([logits_0100, logits_1011], dim=0)
#         print(logits.size())
#         # repeat twice to match shape of logits
#         labels = labels.repeat(2)
#         loss = self.cross_entropy(logits, labels)
#
#         return loss + ne_loss
#
#
# class NTXentLoss():
#     def __init__(self, temperature: float = 0.5):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
#         self.eps = 1e-8
#
#         if abs(self.temperature) < self.eps:
#             raise ValueError('Illegal temperature: abs({}) < 1e-8'
#                              .format(self.temperature))
#
#     def forward(self, out0: torch.Tensor, out1: torch.Tensor):
#
#         device = out0.device
#         batch_size, _ = out0.shape
#
#         # normalize the output to length 1
#         out0 = torch.nn.functional.normalize(out0, dim=1)
#         out1 = torch.nn.functional.normalize(out1, dim=1)
#
#         # out1, negatives = super(NTXentLoss, self).forward(out1, update=out0.requires_grad)
#
#         out0_large = out0
#         out1_large = out1
#
#         logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
#         logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
#         logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
#         logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature
#
#         # initialize labels and masks
#         labels = torch.arange(batch_size, device=device, dtype=torch.long)
#         masks = torch.ones_like(logits_00).bool()
#         masks.scatter_(dim=1, index=labels.unsqueeze(1), value=False)
#
#         # remove similarities of samples to themselves
#         logits_00 = logits_00[masks].view(batch_size, -1)
#         logits_11 = logits_11[masks].view(batch_size, -1)
#
#         # concatenate logits
#         # the logits tensor in the end has shape (2*n, 2*m-1)
#         logits_0100 = torch.cat([logits_01, logits_00], dim=1)
#         logits_1011 = torch.cat([logits_10, logits_11], dim=1)
#         logits = torch.cat([logits_0100, logits_1011], dim=0)
#
#         # repeat twice to match shape of logits
#         labels = labels.repeat(2)
#
#         loss = self.cross_entropy(logits, labels)
#
#         return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)  #

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        # labels = torch.arange(self.class_num, device=positive_clusters.device, dtype=torch.long)
        # labels = labels.repeat(2)
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        # loss /= N

        return loss + ne_loss


class DCLoss(nn.Module):

    def __init__(self, lamda=0.5):
        super(DCLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.t = lamda

    def forward(self, x, x_tf):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == x_tf.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        # pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        # loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).to(cfg.device))

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()

        # t = 0.5

        x_norm = F.normalize(x)
        x_tf_norm = F.normalize(x_tf)

        logits = torch.mm(x_norm, x_tf_norm.t()) / self.t

        labels = torch.tensor(range(logits.shape[0])).cuda()

        # for c
        x_norm = F.normalize(x, dim=0)
        x_tf_norm = F.normalize(x_tf, dim=0)
        logits_c = torch.mm(x_norm.t(), x_tf_norm) / self.t

        labels_c = torch.tensor(range(logits_c.shape[0])).cuda()

        loss = torch.nn.CrossEntropyLoss()(logits, labels) + torch.nn.CrossEntropyLoss()(logits_c, labels_c) + loss_ne

        # loss1 = torch.nn.CrossEntropyLoss()(logits, labels)

        # loss2 = torch.nn.CrossEntropyLoss()(logits_c, labels_c)

        return loss


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    sim_data_1 = autograd.Variable(torch.randn(16, 16))
    sim_data_2 = autograd.Variable(torch.randn(16, 16))
    cl = ClusterLoss(16, 1, torch.device('cuda'))
