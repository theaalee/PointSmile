from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import sklearn.metrics as metrics
import utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import ModelNet
from model import PointNetfeat, BYOL, DGCNN, PointNet2, ShellNet,DGCNN_cls
import datetime
from pathlib import Path
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler import CosineLRScheduler
import pytorch_warmup as warmup


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    return parser.parse_args()


def little_test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

# 92.7
# 92.9


def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    trainDataLoader = DataLoader(ModelNet(partition='train', num_points=1024), num_workers=8,
                              batch_size=32, shuffle=True, drop_last=True)
    testDataLoader = DataLoader(ModelNet(partition='test', num_points=1024), num_workers=8,
                             batch_size=32, shuffle=True, drop_last=False)

    '''MODEL LOADING'''
    num_class = args.num_category
    classifier = DGCNN_cls(args).to()
    checkpoint = torch.load('/home/lx/桌面/best_dgcnn_model.pth')
        # start_epoch = checkpoint['epoch']
    # classifier = utils.copy_parameters(classifier, checkpoint, verbose=True)
    classifier.load_state_dict(checkpoint, strict=False)
    classifier.cuda()
    log_string('Use pretrain model')

    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         classifier.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.decay_rate
    #     )
    # else:
    #     optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    #
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # optimizer = optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=5e-2)
    # scheduler = CosineLRScheduler(optimizer, t_initial=300, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=10)
    fc_params_id = list(map(id, classifier.classification_head.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params_id, classifier.parameters())
    optimizer = optim.SGD([{'params': base_params, 'lr': 0.01},
                           {'params': classifier.classification_head.parameters(), 'lr': 0.001 * 100}], momentum=0.9, weight_decay=1e-4)

    # optimizer = optim.SGD(classifier.parameters(), lr=0.001 * 100, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, 300, eta_min=0.001,)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    print('Start training...')
    best_test_acc = 0
    for epoch in range(0, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        criterion = utils.get_loss

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # points = utils.random_point_dropout(points)
            # points[:, :, 0:3] = utils.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = utils.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points).float()
            # points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda().squeeze()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            scheduler.step(epoch=batch_id)

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # with torch.no_grad():
            # instance_acc, class_acc = little_test(classifier.eval(), testDataLoader, num_class=num_class)
            #
            # if (instance_acc >= best_instance_acc):
            #     best_instance_acc = instance_acc
            #     best_epoch = epoch + 1
            #
            # if (class_acc >= best_class_acc):
            #     best_class_acc = class_acc
            # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            # log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            #
            # if (instance_acc >= best_instance_acc):
            #     logger.info('Save model...')
            #     savepath = str(checkpoints_dir) + '/best_model.pth'
            #     log_string('Saving at %s' % savepath)
            #     state = {
            #         'epoch': best_epoch,
            #         'instance_acc': instance_acc,
            #         'class_acc': class_acc,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
        test_loss = 0.0
        count = 0.0
        classifier.eval()
        test_pred = []
        test_true = []
        for data, label in testDataLoader:
            data, label = data.cuda(), label.cuda().squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = classifier(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
        if avg_per_class_acc >= best_instance_acc:
            best_instance_acc = avg_per_class_acc
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        print(outstr)

        outstr = 'best test acc: %.6f, best test avg acc: %.6f' % (best_test_acc, best_instance_acc)
        print(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(classifier.state_dict(), 'checkpoints/exp/models/model.t7')

        global_epoch += 1

    logger.info('End of training...')


if __name__ == "__main__":
    args = parse_args()
    main(args)