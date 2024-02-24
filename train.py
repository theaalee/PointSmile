from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
from sklearn.svm import SVC
import torch.optim as optim
# from loss import ClusterLoss, InstanceLoss, DCLoss
from torch.utils.data import DataLoader
from data import ShapeNetDatasetCpt, ModelNet40SVM, ModelNet40SVMPretrain, ShapeNet
from model import PointNetfeat, BYOL, DGCNN, PointNet2, ShellNet
from utils.utils import IOStream, AverageMeter
import torchvision.transforms as transforms
import utils.data_utils as d_utils


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    wandb.init(project="PointSmile", name=args.exp_name)
    # DATASET_PATH = 'shapenetcore_partanno_segmentation_benchmark_v0'
    # ShapenetDataset = ShapeNetDatasetCpt(root=DATASET_PATH, n_imgs=2048)
    ModelNet40Dataset = ModelNet40SVMPretrain(num_points=1024, partition='train')
    # ShapenetDataset = ShapeNet(DATA_PATH='ShapeNet55-34/ShapeNet-55', PC_PATH='/home/lx/桌面/ShapeNet55-34/shapenet_pc', subset='train', N_POINTS=1024)
    train_loader = DataLoader(ModelNet40Dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # train_loader = DataLoader(ModelNet40SVMPretrain(partition='train', num_points=1024), num_workers=4, batch_size=16, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        point_model = DGCNN(args).to(device)
    elif args.model == 'dgcnn_seg':
        point_model = DGCNN_partseg(args).to(device)
    elif args.model == 'pointnet':
        point_model = PointNetfeat().to(device)
        # point_model = DGCNN().to(device)
        # point_model = PointNet2().to(device)
        # point_model = ShellNet(num_points=2048, num_class=2).to(device)
    else:
        raise Exception("Not implemented")

    wandb.watch(point_model)

    print("Use SGD")
    opt = optim.SGD(point_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature=0.1).to(device)
    # criterion = InstanceLoss(0.5, torch.device('cuda')).cuda()
    # criterion_cluster = ClusterLoss(16, 1.0, torch.device('cuda')).cuda()

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()

        train_loader.epoch = epoch
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        train_imid_losses = AverageMeter()
        train_cluster_losses = AverageMeter()

        point_model.train()

        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, (data_t1, data_t2) in enumerate(train_loader):
            # for i, data in enumerate(train_loader):

            data_t1, data_t2 = data_t1.to(device), data_t2.to(device)
            batch_size = data_t1.size()[0]

            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()

            _, _, point_feats, point_cluster = point_model(data)

            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]

            point_t1_cluster = point_cluster[:batch_size, :]
            point_t2_cluster = point_cluster[batch_size:, :]

            loss_imid = criterion(point_t1_feats, point_t2_feats)
            loss_cluster = criterion(point_t1_cluster.T, point_t2_cluster.T)

            total_loss = loss_imid + loss_cluster
            total_loss.backward()
            opt.step()

            train_losses.update(total_loss.item(), batch_size)
            train_imid_losses.update(loss_imid.item(), batch_size)
            train_cluster_losses.update(loss_cluster.item(), batch_size)

            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f' % (
                epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg))
                # print('Epoch (%d), Batch(%d/%d), loss: %.6f' % (epoch, i, len(train_loader), losses.avg))

        # ShapenetDataset.add_epoch()
        # wandb_log['Train Loss'] = losses.avg
        wandb_log['Train IMID Loss'] = train_imid_losses.avg
        wandb_log['Train CLUSTER Loss'] = train_cluster_losses.avg
        #
        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        # outstr = 'Train %d, loss: %.6f' % (epoch, losses.avg)
        io.cprint(outstr)

        # Testing

        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=93, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=92, shuffle=True)

        feats_train = []
        labels_train = []
        point_model.eval()
        # byol.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            # data = data.to(device)
            with torch.no_grad():
                feats, _, _, _ = point_model(data)
                # feats = byol(data, isFeat=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            # data = data.to(device)
            with torch.no_grad():
                feats, _, _, _ = point_model(data)
                # feats = byol(data, isFeat=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        model_tl = SVC(C=0.1, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

        wandb.log(wandb_log)

    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model.state_dict(), save_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=80, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    # else:
    #     test(args, io)