# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import faiss
import torch.autograd as autograd
from loss import entropy
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from data import NeighborsDataset
from sys import byteorder


def modelnet_cat2num(modelnet_root):
    for i, item in enumerate(os.listdir(modelnet_root)):
        with open(os.path.join(modelnet_root, 'modelnet_cat2num.txt'), 'a') as f:
            f.write(item + ' ' + str(i) + '\n')


shapenet_labels = {'Airplane': 4,
                   'Bag': 2,
                   'Cap': 2,
                   'Car': 4,
                   'Chair': 4,
                   'Earphone': 3,
                   'Guitar': 3,
                   'Knife': 2,
                   'Lamp': 4,
                   'Laptop': 2,
                   'Motorbike': 6,
                   'Mug': 2,
                   'Pistol': 3,
                   'Rocket': 3,
                   'Skateboard': 3,
                   'Table': 3
                   }


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def s3d_cat2num(s3d_root):
    for item in s3d_cat2num:
        with open(os.path.join(s3d_root, 's3d_cat2num.txt'), 'a') as f:
            f.write(item + ' ' + str(s3d_cat2num[item]) + '\n')


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.
    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1, -1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, self.C),
                                    yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatL2(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk + 1)  # Sample itself is included
        # print(indices, distances)
        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)
        targets = targets.reshape(-1)
        assert (b + self.ptr <= self.n)
        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i,  (data, label, _) in enumerate(loader):
        data = data.type(torch.DoubleTensor)
        data = data.permute(0, 2, 1).cuda()
        targets = label.cuda(non_blocking=True)
        output, _, _ = model(data)
        memory_bank.update(output, targets)
        if i % 10 == 0:
            print('Fill Memory Bank [%d/%d]' % (i, len(loader)))


@torch.no_grad()
def get_predictions(dataloader, model, return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(1)]  # p['num_heads']
    probs = [[] for _ in range(1)]  # p['num_heads']
    targets = []
    if return_features:
        ft_dim = 1024
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        images = images.permute(0, 2, 1)
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['possible_neighbors'])

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for
               pred_, prob_ in zip(predictions, probs)]
    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None,
                       compute_purity=True, compute_confusion_matrix=True,
                       confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    targets = targets.reshape(-1)
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


import os, logging, datetime, random
from pathlib import Path


class TrainLogger:

    def __init__(self, args, name='model', subfold='cls', filename='train_log', cls2name=None):
        self.step = 1
        self.epoch = 1
        self.args = args
        self.name = name
        self.sf = subfold
        self.mkdir()
        self.setup(filename=filename)
        self.epoch_init()
        self.save_model = False
        self.cls2name = cls2name
        self.best_instance_acc, self.best_class_acc, self.best_miou = 0., 0., 0.
        self.best_instance_epoch, self.best_class_epoch, self.best_miou_epoch = 0, 0, 0
        self.savepath = str(self.checkpoints_dir) + '/best_model.pth'

    def setup(self, filename='train_log'):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, filename + '.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # ref: https://stackoverflow.com/a/53496263/12525201
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # logging.getLogger('').addHandler(console) # this is root logger
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)
        self.logger.removeHandler(console)

    def mkdir(self):
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./log/')
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath(self.sf)
        experiment_dir.mkdir(exist_ok=True)

        if self.args.log_dir is None:
            self.experiment_dir = experiment_dir.joinpath(timestr)
        else:
            self.experiment_dir = experiment_dir.joinpath(self.args.log_dir)

        self.experiment_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.log_dir = self.experiment_dir.joinpath('logs/')
        self.log_dir.mkdir(exist_ok=True)

    # @property.setter
    def epoch_init(self, training=True):
        self.loss, self.count, self.pred, self.gt = 0., 0., [], []
        if training:
            self.logger.info('Epoch %d/%d:' % (self.epoch, self.args.epoch))

    def step_update(self, pred, gt, loss, training=True):
        if training:
            self.step += 1  # Use TensorFlow way to count training steps
        self.gt.append(gt)
        self.pred.append(pred)
        batch_size = len(pred)
        self.count += batch_size
        self.loss += loss * batch_size

    def epoch_update(self, training=True, mode='cls'):
        self.save_model = False
        self.gt = np.concatenate(self.gt)
        self.pred = np.concatenate(self.pred)

        instance_acc = metrics.accuracy_score(self.gt, self.pred)
        if instance_acc > self.best_instance_acc and not training:
            self.save_model = True if mode == 'cls' else False
            self.best_instance_acc = instance_acc
            self.best_instance_epoch = self.epoch

        if mode == 'cls':
            class_acc = metrics.balanced_accuracy_score(self.gt, self.pred)
            if class_acc > self.best_class_acc and not training:
                self.best_class_epoch = self.epoch
                self.best_class_acc = class_acc
            return instance_acc, class_acc
        elif mode == 'semseg':
            miou = self.calculate_IoU().mean()
            if miou > self.best_miou and not training:
                self.best_miou_epoch = self.epoch
                self.save_model = True
                self.best_miou = miou
            return instance_acc, miou
        elif mode == 'partseg':
            miou = self.calculate_IoU().mean()
            if miou > self.best_miou and not training:
                self.best_miou_epoch = self.epoch
                self.save_model = True
                self.best_miou = miou
                print(instance_acc, miou)
            return instance_acc, miou
        else:
            raise ValueError('Mode is not Supported by TrainLogger')

    def epoch_summary(self, writer=None, training=True, mode='cls'):
        criteria = 'Class Accuracy' if mode == 'cls' else 'mIoU'
        instance_acc, class_acc = self.epoch_update(training=training, mode=mode)
        if training:
            if writer is not None:
                writer.add_scalar('Train Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Train %s' % criteria, class_acc, self.step)
            self.logger.info('Train Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Train %s: %.3f' % (criteria, class_acc))
        else:
            if writer is not None:
                writer.add_scalar('Test Instance Accuracy', instance_acc, self.step)
                writer.add_scalar('Test %s' % criteria, class_acc, self.step)
            self.logger.info('Test Instance Accuracy: %.3f' % instance_acc)
            self.logger.info('Test %s: %.3f' % (criteria, class_acc))
            self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
                self.best_instance_acc, self.best_instance_epoch))
            if self.best_class_acc > .1:
                self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                    self.best_class_acc, self.best_class_epoch))
            if self.best_miou > .1:
                self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                    self.best_miou, self.best_miou_epoch))

        self.epoch += 1 if not training else 0
        if self.save_model:
            self.logger.info('Saving the Model Params to %s' % self.savepath)

    def calculate_IoU(self):
        num_class = len(self.cls2name)
        Intersection = np.zeros(num_class)
        Union = Intersection.copy()
        # self.pred -> numpy.ndarray (total predictions, )

        for sem_idx in range(num_class):
            Intersection[sem_idx] = np.sum(np.logical_and(self.pred == sem_idx, self.gt == sem_idx))
            Union[sem_idx] = np.sum(np.logical_or(self.pred == sem_idx, self.gt == sem_idx))
        return Intersection / Union

    def train_summary(self, mode='cls'):
        self.logger.info('\n\nEnd of Training...')
        self.logger.info('Best Instance Accuracy: %.3f at Epoch %d ' % (
            self.best_instance_acc, self.best_instance_epoch))
        if mode == 'cls':
            self.logger.info('Best Class Accuracy: %.3f at Epoch %d' % (
                self.best_class_acc, self.best_class_epoch))
        elif mode == 'semseg':
            self.logger.info('Best mIoU: %.3f at Epoch %d' % (
                self.best_miou, self.best_miou_epoch))

    def update_from_checkpoints(self, checkpoint):
        self.logger.info('Use Pre-Trained Weights')
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_instance_epoch, self.best_instance_acc = checkpoint['epoch'], checkpoint['instance_acc']
        self.best_class_epoch, self.best_class_acc = checkpoint['best_class_epoch'], checkpoint['best_class_acc']
        self.logger.info('Best Class Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_class_epoch))
        self.logger.info('Best Instance Acc {:.3f} at Epoch {}'.format(self.best_instance_acc, self.best_instance_epoch))


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU Usage
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def copy_parameters(model, pretrained, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    model_dict = model.state_dict()
    pretrained_dict = pretrained
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def weights_init(m):
    """
    Xavier normal initialisation for weights and zero bias,
    find especially useful for completion and segmentation Tasks
    """
    classname = m.__class__.__name__
    if (classname.find('Conv1d') != -1) or (classname.find('Conv2d') != -1) or (classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def _open_stream(stream, read_or_write):
    if hasattr(stream, read_or_write):
        return (False, stream)
    try:
        return (True, open(stream, read_or_write[0] + 'b'))
    except TypeError:
        raise RuntimeError("expected open file or filename")


if __name__ == '__main__':
    '''
    modelnet_root = './ModelNet10'
    modelnet_cat2num(modelnet_root)
    s3d_root = './Stanford3dDataset_v1.2'
    s3d_cat2num(s3d_root)
    '''
    # output = autograd.Variable(torch.randn(128, 1024))
    # print(output)
    # targets = autograd.Variable(torch.randn(128, 1))
    # memory_bank = MemoryBank(128, 1024, 40, 0.1)
    # memory_bank.update(output, targets)
    # memory_bank.mine_nearest_neighbors(5)
    d = 64
    nb = 100
    nq = 10
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    output = autograd.Variable(torch.from_numpy(xb))
    targets = autograd.Variable(torch.randn(100, 1))
    memory_bank = MemoryBank(100, 64, 40, 0.1)
    memory_bank.update(output, targets)
    memory_bank.mine_nearest_neighbors(5)

    # print(xb[:2])
    # xb[:, 0] += np.arange(nb).astype('float32') / 1000
    # sys.exit()
    # print(xb[:2])
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq).astype('float32') / 1000
    index = faiss.IndexFlatL2(d)  # buid the index
    index.add(xb)
    print(index.ntotal)

    k = 4
    D, I = index.search(xb[:5], k)
    print("IIIIIIIIIIII")
    print(I)
    print("ddddddddd")
    print(D)

