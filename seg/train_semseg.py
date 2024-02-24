import numpy as np
import os, sys, torch, shutil, argparse, importlib
sys.path.append('utils')
sys.path.append('models')
from utils import cal_loss, IOStream
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn as nn
from data import S3DISDataset_HDF5, S3DIS
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DGCNN_semseg, get_loss
from plyfile import PlyElement, PlyData
import sklearn.metrics as metrics


classes = ['ceiling', 'floor', 'wall', 'beam', 'column',
           'window', 'door', 'table', 'chair', 'sofa',
           'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    parser.add_argument('--log_dir', type=str, help='log path [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size [default: 24]')
    # parser.add_argument('--test_area', type=int, default=5, help='test area, 1-6 [default: 5]')\
    parser.add_argument('--test_area', type=str, default='6', metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--epoch', default=100, type=int, help='training epochs [default: 100]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum [default: 0.9]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--restore', default=True, action='store_true', help='restore the weights [default: False]')
    parser.add_argument('--restore_path', type=str, help='path to pre-saved model weights [default: ]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate in FCs [default: 0.5]')
    parser.add_argument('--bn_decay', action='store_true', help='use BN Momentum Decay [default: False]')
    parser.add_argument('--xavier_init', action='store_true', help='Xavier weight init [default: False]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='embedding dimensions [default: 1024]')
    parser.add_argument('--k', type=int, default=20, help='num of nearest neighbors to use [default: 20]')
    parser.add_argument('--step_size', type=int, default=40, help='lr decay steps [default: every 40 epochs]')
    parser.add_argument('--scheduler', type=str, default='cos', help='lr decay scheduler [default: cos, step]')
    parser.add_argument('--model', type=str, default='dgcnn_semseg', help='model [default: pointnet_semseg]')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimiser [default: adam, otherwise sgd]')
    parser.add_argument('--exp_name', type=str, default='sem', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--visu', type=str, default='area_6',
                        help='visualize the model')
    return parser.parse_args()


def train(args, io):
    root = '/home/lx/桌面/indoor3d_sem_seg_hdf5_data'
    TRAIN_DATASET = S3DISDataset_HDF5(root=root, split='train', test_area=args.test_area)
    TEST_DATASET = S3DISDataset_HDF5(root=root, split='test', test_area=args.test_area)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model_%s.t7' % (args.exp_name, args.test_area))


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def visualization(visu, visu_format, test_choice, data, seg, pred, visual_file_index, semseg_colors):
    global room_seg, room_pred
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index + 1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != \
                        roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False
            else:
                skip = True
        elif test_choice != 'all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists(
                    'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname):
                os.makedirs(
                    'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname)

            data = np.loadtxt(
                'prepare_data/data/indoor3d_sem_seg_hdf5_data_test/raw_data3d/Area_' + test_area + '/' + roomname + '(' + str(
                    visual_file_index) + ').txt')
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB.append(semseg_colors[int(pred[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:, [1, 2, 0]]
            xyzRGB = np.concatenate((data, np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_seg.append(seg[i].cpu().numpy())
            room_pred.append(pred[i].cpu().numpy())
            f = open(
                'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt',
                "a")
            f_gt = open(
                'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt',
                "a")
            np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ')
            np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ')

            if roomname != roomname_next:
                mIoU = np.nanmean(calculate_sem_IoU(np.array(room_pred), np.array(room_seg)))
                mIoU = str(round(mIoU, 4))
                room_pred = []
                room_seg = []
                if visu_format == 'ply':
                    filepath = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_pred_' + mIoU + '.ply'
                    filepath_gt = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.ply'
                    xyzRGB = np.loadtxt(
                        'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt')
                    xyzRGB_gt = np.loadtxt(
                        'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt')
                    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i
                              in range(xyzRGB.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4],
                                  xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(xyzRGB,
                                                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath)
                    print('PLY visualization file saved in', filepath)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt,
                                                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                                                                 ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system(
                        'rm -rf ' + 'outputs/' + args.exp_name + '/visualization/area_' + test_area + '/' + roomname + '/*.txt')
                else:
                    filename = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '.txt'
                    filename_gt = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_gt.txt'
                    filename_mIoU = 'outputs/' + args.exp_name + '/' + 'visualization' + '/' + 'area_' + test_area + '/' + roomname + '/' + roomname + '_pred_' + mIoU + '.txt'
                    os.rename(filename, filename_mIoU)
                    print('TXT visualization file saved in', filename_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                      (visu_format))
                exit()


def vis(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1, 7):
        visual_file_index = 0
        test_area = str(test_area)
        if os.path.exists("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
                    visual_file_index = visual_file_index + 1
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=4096, test_area=test_area),
                                     batch_size=8, shuffle=False, drop_last=False)

            device = torch.device("cuda")

            # Try to load models
            semseg_colors = test_loader.dataset.semseg_colors
            model = DGCNN_semseg(args).to(device)

            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join('/home/lx/桌面', 'model_%s.t7' % test_area)))
            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # visiualization
                visualization(args.visu, args.visu_format, args.test_area, data, seg, pred, visual_file_index,
                              semseg_colors)
                visual_file_index = visual_file_index + data.shape[0]
            if visual_warning and args.visu != '':
                print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == '__main__':
    args = parse_args()
    io = IOStream('outputs/exp' + '/run.log')
    io.cprint(str(args))
    # main(args)
    vis(args, io)
    # visualization('', 'ply', args.test_area, data, seg, pred, visual_file_index, semseg_colors)