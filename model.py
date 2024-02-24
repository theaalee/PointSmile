import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from lightly.loss.ntx_ent_loss import NTXentLoss


class STN3D(nn.Module):
    def __init__(self):
        super(STN3D, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        # if torch.any(torch.isnan(x)):
        #     for parameters in self.conv1.parameters():
        #         print(parameters)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = autograd.Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1,9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x += iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3D()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 512)
        # self.bn4 = nn.BatchNorm1d(512)
        # add mlp projection head
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # x = batch,64,n

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024

        if self.global_feat:
            # x = F.relu(self.bn4(self.fc1(x)))  # x = batch,512
            # x = self.fc2(x)
            x_p = F.normalize(self.projection_head(x), dim=1)
            x_c = self.cluster_projector(x)
            return x, trans, x_p, x_c
        else:
            # x = F.relu(self.bn4(self.fc1(x)))  # x = batch,512
            # x = self.fc2(x)
            # x_p = F.normalize(self.projection_head(x), dim=1)
            # x_c = self.cluster_projector(x)
            # x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, shallow_feat], 1), trans#, x_p, x_c


class PointNetCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0), trans


class PointNetSeg(nn.Module):
    def __init__(self, k=2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, label=1):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)

        return x, trans


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, cls=-1):
        super(DGCNN, self).__init__()
        self.k = 20
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        if cls != -1:
            self.linear1 = nn.Linear(512 * 2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=0.5)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=0.5)
            self.linear3 = nn.Linear(256, 1024)

        self.cls = cls

        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        feat = x
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)

        x_p = self.projection_head(feat)
        x_c = self.cluster_projector(feat)
        return feat, x, x_p, x_c


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # for p in self.parameters():
        #     p.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, output_channels),
        )
        # self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        x = self.classification_head(x)
        return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetPartSegEncoder(nn.Module):
    def __init__(self, feature_transform=True, channel=3):
        super(PointNetPartSegEncoder, self).__init__()
        self.stn = STN3D()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)
        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.feature_transform:
            trans_feat = self.fstn(out3)
            net_transformed = torch.bmm(out3.transpose(2, 1), trans_feat)
            out3 = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.bn5(self.conv5(out4))

        out_max = torch.max(out5, 2, keepdim=False)[0]
        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        if self.feature_transform:
            return concat, trans_feat
        return


class PointNetPartSeg(nn.Module):
    def __init__(self, part_num=50, normal_channel=True):
        super(PointNetPartSeg, self).__init__()

        self.part_num = part_num
        self.feat = PointNetPartSegEncoder(feature_transform=True)
        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        concat, trans_feat = self.feat(point_cloud, label)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net).transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)  # [B, N, 50]

        return net, trans_feat


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all=None, pretrain=True):
        # def __init__(self, args):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.pretrain = pretrain
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.inv_head = nn.Sequential(
            nn.Linear(args.emb_dims, args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(args.emb_dims, 256)
        )

        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=args.dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       self.bn9,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=args.dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn10,
                                        nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        if self.pretrain:
            print("Pretrain")
            x = x.squeeze()
            inv_feat = self.inv_head(x)

            return x, inv_feat, x

        else:
            l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
            l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
            x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

            x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

            x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

            return x


class DGCNN_semseg(nn.Module):

    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def cal_loss(pred, gold, smoothing=False):
        """Calculate cross entropy loss, apply label smoothing if needed."""

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size()[1]
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

    def forward(self, pred, target):

        return self.cal_loss(pred, target, smoothing=False)


class Augmentor(nn.Module):
    def __init__(self, dim=1024, in_dim=3):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.rot = Augmentor_Rotation(self.dim)
        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, pt, noise):

        B, C, N = pt.size()
        raw_pt = pt[:, :3, :].contiguous()
        normal = pt[:, 3:, :].transpose(1, 2).contiguous() if C > 3 else None

        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        feat_r = x.view(-1, 1024)
        feat_r = torch.cat([feat_r,noise],1)
        rotation, scale = self.rot(feat_r)

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d,noise_d],1)
        displacement = self.dis(feat_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0
        if p1 > possi:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()
        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt,normal],1)

        return pt


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNet2(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, num_class)

        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Softmax(dim=1)
        )

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        x_p = F.normalize(self.projection_head(x), dim=1)
        x_c = self.cluster_projector(x)
        return x, _, x_p, x_c
        #
        # return x,l3_points


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp = False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.hidden = {}
        self.hook_registered = False

    # def _find_layer(self):
    #     if type(self.layer) == str:
    #         modules = dict([*self.net.named_modules()])
    #         return modules.get(self.layer, None)
    #     elif type(self.layer) == int:
    #         children = [*self.net.children()]
    #         return children[self.layer]
    #     return None
    #
    # def _hook(self, _, input, output):
    #     device = input[0].device
    #     self.hidden[device] = flatten(output)
    #
    # def _register_hook(self):
    #     layer = self._find_layer()
    #     assert layer is not None, f'hidden layer ({self.layer}) not found'
    #     handle = layer.register_forward_hook(self._hook)
    #     self.hook_registered = True
    #
    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(get_module_device(self.net))
    #
    # def get_representation(self, x):
    #     if self.layer == -1:
    #         return self.net(x)
    #
    #     # if not self.hook_registered:
    #     #     self._register_hook()
    #
    #     self.hidden.clear()
    #     hidden, _ = self.net(x)
    #
    #     # hidden = self.hidden
    #     # self.hidden = None
    #
    #     assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
    #     return hidden

    def forward(self, x, return_projection=True):
        representation, _ = self.net(x)

        # if not return_projection:
        #     return representation
        projector = self._get_projector(representation)
        # _, dim = representation.shape
        # projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        projection = projector(representation)
        return projection, representation


class BYOL(nn.Module):
    def __init__(self, net, hidden_layer=-2, projection_size=256, projection_hidden_size=4096, moving_average_decay=0.99, use_momentum=True, m = 0.9):
        super().__init__()
        self.net = net
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.m = m
        self.criterion = NTXentLoss(temperature=0.1)
        self.cluster_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Softmax(dim=1)
        )

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters

    @singleton('target_encoder')
    def _get_target_encoder(self):
        import copy
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        # for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
        #     param_k.data = param_q.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x, isFeat=False):
        assert not (self.training and x.shape[0] == 1), \
            'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        batch_size = int(x.size()[0]/2)

        if isFeat:
            online_proj, features = self.online_encoder(x)
            return features

        online_proj, features = self.online_encoder(x)
        # online_proj_one = online_proj[:batch_size, :]
        # online_proj_two = online_proj[batch_size:, :]

        online_pred = self.online_predictor(online_proj)
        online_pred_one = online_pred[:batch_size, :]
        online_pred_two = online_pred[batch_size:, :]


        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj, _ = target_encoder(x)
            target_proj_one = target_proj[:batch_size, :]
            target_proj_two = target_proj[batch_size:, :]
            target_proj_one.detach()
            target_proj_two.detach()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        point_cluster = self.cluster_projector(features)
        point_t1_cluster = point_cluster[:batch_size, :]
        point_t2_cluster = point_cluster[batch_size:, :]

        loss_cluster = self.criterion(point_t1_cluster.T, point_t2_cluster.T)
        loss = loss_one + loss_two
        return loss.mean() + loss_cluster

#
# def knn(points, queries, K):
#     """
#     Args:
#         points ( B x N x 3 tensor )
#         query  ( B x M x 3 tensor )  M < N
#         K      (constant) num of neighbors
#     Outputs:
#         knn    (B x M x K x 3 tensor) sorted K nearest neighbor
#         indice (B x M x K tensor) knn indices
#     """
#     value = None
#     indices = None
#     num_batch = points.shape[0]
#     for i in range(num_batch):
#         point = points[i]
#         query = queries[i]
#         dist = torch.cdist(point, query)
#         idxs = dist.topk(K, dim=0, largest=False, sorted=True).indices
#         idxs = idxs.transpose(0, 1)
#         nn = point[idxs].unsqueeze(0)
#         value = nn if value is None else torch.cat((value, nn))
#
#         idxs = idxs.unsqueeze(0)
#         indices = idxs if indices is None else torch.cat((indices, idxs))
#
#     return value.long(), indices.long()


def gather_feature(features, indices):
    """
    Args:
        features ( B x N x F tensor) -- feature from previous layer
        indices  ( B x M x K tensor) --  represents queries' k nearest neighbor
    Output:
        features ( B x M x K x F tensor) -- knn features from previous layer
    """
    res = None
    num_batch = features.shape[0]
    for B in range(num_batch):
        knn_features = features[B][indices[B]].unsqueeze(0)
        res = knn_features if res is None else torch.cat((res, knn_features))
    return res


def random_sample(points, num_sample):
    """
    Args:
        points ( B x N x 3 tensor )
        num_sample (constant)
    Outputs:
        sampled_points (B x num_sample x 3 tensor)
    """
    perm = torch.randperm(points.shape[1])
    return points[:, perm[:num_sample]].clone()


class Dense(nn.Module):
    def __init__(self, in_size, out_size, in_dim=3,
                 has_bn=True, drop_out=None):
        super(Dense, self).__init__()
        """
        Args:
            input ( B x M x K x 3  tensor ) -- subtraction vectors 
                from query to its k nearest neighbor
        Output: 
            local point feature ( B x M x K x 64 tensor ) 
        """
        self.has_bn = has_bn
        self.in_dim = in_dim

        if in_dim == 3:
            self.batchnorm = nn.BatchNorm1d(in_size)
        elif in_dim == 4:
            self.batchnorm = nn.BatchNorm2d(in_size)
        else:
            self.batchnorm = None

        if drop_out is None:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            )

    def forward(self, inputs):

        if self.has_bn == True:
            d = self.in_dim - 1
            outputs = self.batchnorm(inputs.transpose(1, d)).transpose(1, d)
            outputs = self.linear(outputs)
            return outputs

        else:
            outputs = self.linear(inputs)
            return outputs


class ShellConv(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division,
                 has_bn=True):
        super(ShellConv, self).__init__()
        """
        out_features  (int) num of output feature (dim = -1)
        prev_features (int) num of prev feature (dim = -1)
        neighbor      (int) num of nearest neighbor in knn
        division      (int) num of division
        """

        self.K = neighbor
        self.S = int(self.K / division)  # num of feaure per shell
        self.F = 64  # num of local point features
        self.neighbor = neighbor
        in_channel = self.F + prev_features
        out_channel = out_features

        self.dense1 = Dense(3, self.F // 2, in_dim=4, has_bn=has_bn)
        self.dense2 = Dense(self.F // 2, self.F, in_dim=4, has_bn=has_bn)
        self.maxpool = nn.MaxPool2d((1, self.S), stride=(1, self.S))
        if has_bn == True:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )

    def forward(self, points, queries, prev_features):
        """
        Args:
            points          (B x N x 3 tensor)
            query           (B x M x 3 tensor) -- note that M < N
            prev_features   (B x N x F1 tensor)
        Outputs:
            feat            (B x M x F2 tensor)
        """
        knn_pts, idxs = knn(points, queries, self.K)
        knn_center = queries.unsqueeze(2)
        knn_points_local = knn_center - knn_pts

        knn_feat_local = self.dense1(knn_points_local)
        knn_feat_local = self.dense2(knn_feat_local)

        # shape: B x M x K x F
        if prev_features is not None:
            knn_feat_prev = gather_feature(prev_features, idxs)
            knn_feat_cat = torch.cat((knn_feat_local, knn_feat_prev), dim=-1)
        else:
            knn_feat_cat = knn_feat_local

        knn_feat_cat = knn_feat_cat.permute(0, 3, 1, 2)  # BMKF -> BFMK
        knn_feat_max = self.maxpool(knn_feat_cat)
        output = self.conv(knn_feat_max).permute(0, 2, 3, 1)

        return output.squeeze(2)


class ShellUp(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division,
                 has_bn=True):
        super(ShellUp, self).__init__()
        self.has_bn = has_bn
        self.sconv = ShellConv(out_features, prev_features, neighbor,
                               division, has_bn)
        self.dense = Dense(2 * out_features, out_features, has_bn=has_bn)

    def forward(self, points, queries, prev_features, feat_skip_connect):
        sconv = self.sconv(points, queries, prev_features)
        feat_cat = torch.cat((sconv, feat_skip_connect), dim=-1)

        outputs = self.dense(feat_cat)
        return outputs


class ShellNet(nn.Module):
    def __init__(self, num_class, num_points,
                 conv_scale=1, dense_scale=1, has_bn=True):
        super(ShellNet, self).__init__()
        self.num_points = num_points
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / conv_scale) for x in filters]

        features = [256, 128]
        features = [int(x / dense_scale) for x in features]

        self.shellconv1 = ShellConv(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv(filters[3], filters[2], 8, 1, has_bn)
        self.shellconv4 = ShellConv(filters[4], filters[3], 4, 1, has_bn)

        self.shellup3 = ShellUp(filters[2], filters[3], 8, 1, has_bn)
        self.shellup2 = ShellUp(filters[1], filters[2], 16, 2, has_bn)
        self.shellup1 = ShellConv(filters[0], filters[1], 32, 4, has_bn)

        # self.fc1 = Dense(filters[0], features[0], has_bn=has_bn, drop_out=0)
        # self.fc2 = Dense(features[0], features[1], has_bn=has_bn, drop_out=0.5)
        # self.fc3 = Dense(features[1], num_class, has_bn=has_bn)

        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        query1 = random_sample(inputs, self.num_points // 2)
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, self.num_points // 4)
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, self.num_points // 8)
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)

        query4 = random_sample(query3, self.num_points // 16)
        sconv4 = self.shellconv4(query3, query4, sconv3)
        # print("sconv4.shape = ", sconv4.shape)
        #
        # up3 = self.shellup3(query3, query2, sconv3, sconv2)
        # print("up3.shape = ", up3.shape)
        #
        # up2 = self.shellup2(query2, query1, up3, sconv1)
        # print("up2.shape = ", up2.shape)
        #
        # up1 = self.shellup1(query1, inputs, up2)
        # print("up1.shape = ", up1.shape)

        # fc1 = self.fc1(up1)
        # # print("fc1.shape = ", fc1.shape)
        #
        # fc2 = self.fc2(fc1)
        # # print("fc2.shape = ", fc2.shape)
        #
        # output = self.fc3(fc2)
        # # print("fc3.shape = ", output.shape)
        x = torch.max(sconv4, 1, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        # print(x.shape)
        x_p = F.normalize(self.projection_head(x), dim=1)
        x_c = self.cluster_projector(x)

        return x, x, x_p, x_c


if __name__ == '__main__':
    sim_data = autograd.Variable(torch.randn(32, 4048, 3))
    shellnet = ShellNet(num_points=4048, num_class=2)
    out = shellnet(sim_data)
    print('stn', out.size())
    # pointfeat = FeatAdProj(global_feat=True)
    # out, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=False)
    # out, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    #
    # cls = PointNetCls(k=4)
    # out, _ = cls(sim_data)
    # print('class', out.size())
    #
    # seg = PointNetSeg(k=4)
    # out, _ = seg(sim_data)
    # print('seg', out.size())
    print(out.size())
