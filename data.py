import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import os
import json
import h5py
import utils.data_utils as d_utils
import torchvision.transforms as transforms
import glob
import pickle
from tqdm import tqdm
import cv2
import open3d

# trans_3 = transforms.Compose(
#             [
#                 d_utils.PointcloudToTensor(),
#                 d_utils.PointcloudNormalize(),
#                 d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
#                 d_utils.PointcloudRotate(),
#                 d_utils.PointcloudTranslate(0.5, p=1),
#                 d_utils.PointcloudJitter(p=1),
#                 d_utils.PointcloudRandomInputDropout(p=1),
#                 d_utils.PointcloudRandomCrop(),
#                 d_utils.PointcloudRandomFlip(),
#             ])

trans_3 = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudNormalize(),
    ])


def CAD(epoch):
    # if epoch <= 100:
    #     trans_1 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),  # 82.6
    #             d_utils.PointcloudRotate(),  # 85.0
    #             d_utils.PointcloudTranslate(0.5, p=1),  # 86.4
    #             d_utils.PointcloudJitter(p=1),  # 86.6
    #             d_utils.PointcloudRandomInputDropout(p=1),  # 86.4
    #             d_utils.PointcloudRandomCrop(),
    #         ])
    #
    #     trans_2 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
    #             d_utils.PointcloudRotate(),
    #             d_utils.PointcloudTranslate(0.5, p=1),
    #             d_utils.PointcloudJitter(p=1),
    #             d_utils.PointcloudRandomInputDropout(p=1),
    #             d_utils.PointcloudRandomCrop(),
    #         ])
    #     return trans_1, trans_2
    # elif epoch <= 200:
    #     trans_1 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),  # 82.6
    #             d_utils.PointcloudRotate(),  # 85.0
    #             d_utils.PointcloudTranslate(0.5, p=1),  # 86.4
    #             d_utils.PointcloudJitter(p=1),  # 86.6
    #             d_utils.PointcloudRandomInputDropout(p=1),  # 86.4
    #             d_utils.PointcloudRandomCuboid(),
    #             d_utils.PointcloudSample()
    #         ])
    #
    #     trans_2 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
    #             d_utils.PointcloudRotate(),
    #             d_utils.PointcloudTranslate(0.5, p=1),
    #             d_utils.PointcloudJitter(p=1),
    #             d_utils.PointcloudRandomInputDropout(p=1),
    #             d_utils.PointcloudRandomCrop(),
    #             d_utils.PointcloudSample()
    #         ])
    #     return trans_1, trans_2
    # elif epoch <= 300:
    #     trans_1 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),  # 82.6
    #             d_utils.PointcloudRotate(),  # 85.0
    #             d_utils.PointcloudTranslate(0.5, p=1),  # 86.4
    #             d_utils.PointcloudJitter(p=1),  # 86.6
    #             d_utils.PointcloudRandomInputDropout(p=1),  # 86.4
    #             d_utils.PointcloudSample(),
    #             d_utils.PointCloudShift(),
    #             d_utils.PointcloudRandomCuboid()
    #         ])
    #
    #     trans_2 = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudNormalize(),
    #             d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
    #             d_utils.PointcloudRotate(),
    #             d_utils.PointcloudTranslate(0.5, p=1),
    #             d_utils.PointcloudJitter(p=1),
    #             d_utils.PointcloudRandomInputDropout(p=1),
    #             d_utils.PointcloudSample(),
    #             d_utils.PointCloudShift(),
    #             d_utils.PointcloudRandomCuboid()
    #         ])
    #     return trans_1, trans_2
    p = epoch//100*0.3

    trans_1 = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudNormalize(),
            d_utils.PointcloudScale(lo=0.5, hi=2, p=p),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudTranslate(0.5, p=p),
            d_utils.PointcloudJitter(p=p),
            d_utils.PointcloudRandomInputDropout(p=p),
            d_utils.PointcloudRandomCrop(),
            d_utils.PointcloudRandomFlip()
        ])

    trans_2 = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudNormalize(),
            d_utils.PointcloudScale(lo=0.5, hi=2, p=p),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudTranslate(0.5, p=p),
            d_utils.PointcloudJitter(p=p),
            d_utils.PointcloudRandomInputDropout(p=p),
            d_utils.PointcloudRandomCrop(),
            d_utils.PointcloudRandomFlip()
        ])
    return trans_1, trans_2


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]


class ShapeNet(data.Dataset):
    def __init__(self, DATA_PATH, PC_PATH, subset, N_POINTS):
        self.data_root = DATA_PATH
        self.pc_path = PC_PATH
        self.subset = subset
        self.npoints = N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        choice = np.random.choice(len(data), 20000, replace=True)
        point_set = data[choice, :]

        if self.subset == 'train':
            point_t1 = trans_1(point_set)
            point_t2 = trans_2(point_set)

            point_t1 = point_t1[np.random.choice(len(point_t1), self.npoints, replace=True), :]
            point_t2 = point_t2[np.random.choice(len(point_t2), self.npoints, replace=True), :]

            return point_t1, point_t2

        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()

        return data

    def __len__(self):
        return len(self.file_list)


def load_modelnet_data(partition):
    # BASE_DIR = '/home/lx/桌面'
    DATA_DIR = '/home/lx/桌面'
    all_data = []
    all_label = []
    root = '/home/lx/桌面/modelnet40_ply_hdf5_2048'
    with open(os.path.join(root, '{}_files.txt'.format(partition)), 'r') as f:
        data_files = [os.path.join(root, line.strip().split('/')[-1]) for line in f]

    for h5_name in data_files:
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_shapenet_data():
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath


class ShapeNetDatasetCpt(Dataset):
    def __init__(self, root, n_imgs=1, split='train'):
        self.root = root
        self.data = load_shapenet_data()
        self.n_imgs = n_imgs
        self.split = split
        self.cat2id = {}
        self.seg_classes = {}
        self.epoch = 0
        # parse category file.
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]

        # parse segment num file.
        # with open('misc/num_seg_classes.txt', 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.seg_classes[ls[0]] = int(ls[1])

        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.datapath = []
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        import json
        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat2id.values():
                self.datapath.append([
                    self.id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                    os.path.join(self.root, category, 'points_label', uuid + '.seg')
                ])

        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        # print("classes:", self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # randomly sample self.num_points point from the origin point cloud.
        # choice = np.random.choice(len(seg), 2048, replace=True)
        # point_set = point_set[choice, :]
        # seg = seg[choice]
        choice = np.random.choice(len(point_set), 20000, replace=True)
        point_set = point_set[choice, :]
        trans_1, trans_2 = CAD(self.epoch)

        if self.split == 'train':
            # TODO: why only rotate the x and z axis??

            point_t1 = trans_1(point_set)
            point_t2 = trans_2(point_set)

            point_t1 = point_t1[np.random.choice(len(point_t1), self.n_imgs, replace=True), :]
            point_t2 = point_t2[np.random.choice(len(point_t2), self.n_imgs, replace=True), :]

            return point_t1, point_t2

        if self.split == 'test':

            point_t1 = trans_1(point_set)
            point_t2 = trans_2(point_set)
            choice = np.random.choice(point_set, self.n_imgs, replace=True)
            point_t1 = point_t1[choice, :]
            point_t2 = point_t2[choice, :]
            return point_t1, point_t2

    def __len__(self):
        return len(self.datapath)

    def add_epoch(self):
        self.epoch += 1


def load_color_semseg():
    colors = []
    labels = []
    f = open("misc/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("misc/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


class S3DISDataset_HDF5(Dataset):
    """Chopped Scene"""

    def __init__(self, root='/home/lx/桌面/indoor3d_sem_seg_hdf5_data', split='train', test_area=5):
        self.root = root
        self.all_files = self.getDataFiles(os.path.join(self.root, 'all_files.txt'))
        self.room_filelist = self.getDataFiles(os.path.join(self.root, 'room_filelist.txt'))
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.semseg_colors = load_color_semseg()

        for h5_filename in self.all_files:
            data_batch, label_batch = self.loadh5DataFile(os.path.join('/home/lx/桌面/', h5_filename))
            self.scene_points_list.append(data_batch)
            self.semantic_labels_list.append(label_batch)

        self.data_batches = np.concatenate(self.scene_points_list, 0)
        self.label_batches = np.concatenate(self.semantic_labels_list, 0)

        test_area = 'Area_' + str(test_area)
        train_idxs, test_idxs = [], []

        for i, room_name in enumerate(self.room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        assert split in ['train', 'test']
        if split == 'train':
            self.data_batches = self.data_batches[train_idxs, ...]
            self.label_batches = self.label_batches[train_idxs]
        else:
            self.data_batches = self.data_batches[test_idxs, ...]
            self.label_batches = self.label_batches[test_idxs]

    @staticmethod
    def getDataFiles(list_filename):
        return [line.rstrip() for line in open(list_filename)]

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __getitem__(self, index):
        points = self.data_batches[index, :]
        labels = self.label_batches[index].astype(np.int32)

        return points, labels

    def __len__(self):
        return len(self.data_batches)


class ModelNet40SVMPretrain(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.epoch = 0

    def __getitem__(self, item):
        # point_set = self.data[item][:self.num_points]
        point_set = self.data[item]

        trans_1, trans_2 = CAD(self.epoch)
        point_t1 = trans_1(point_set)
        point_t2 = trans_2(point_set)
        point_t1 = point_t1[np.random.choice(len(point_t1), self.num_points, replace=True), :]
        point_t2 = point_t2[np.random.choice(len(point_t2), self.num_points, replace=True), :]
        return point_t1, point_t2

    def __len__(self):
        return self.data.shape[0]


class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def translate_pointcloud(pointcloud):
    # xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    # xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    #
    # translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')

    return trans_3(pointcloud)


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join('/home/lx/桌面/', 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        # pointcloud = self.data[item][:self.num_points]
        if self.partition == 'train':
            # choice = np.random.choice(len(self.data[item]), 20000, replace=True)
            label = self.label[item]
            pointcloud = self.data[item][:self.num_points]
            pointcloud = translate_pointcloud(pointcloud)
                # pointcloud = trans_1(pointcloud)
                # pointcloud = pointcloud[:self.num_points]
            np.random.shuffle(pointcloud)
        else:
            label = self.label[item]
            pointcloud = self.data[item][:self.num_points]
            # pointcloud = translate_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def load_data_partseg(partition):
    # download_shapenetpart()
    BASE_DIR = '/home/lx/桌面/'
    DATA_DIR = os.path.join(BASE_DIR, '')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, num_category, split='train', process_data=False):
        self.root = '/home/lx/桌面/modelnet40_normal_resampled'  # 数据集根路径
        self.npoints = 1024  # 采样点的数量
        self.process_data = process_data  # 将txt文件转换为.dat文件，仅在第一次读取的时候执行
        self.uniform = False  # 使用FPS（最远点采样）下采样数据
        self.use_normals = False  # 是否使用RGB颜色信息
        self.num_category = num_category  # 选择数据集的类型，如 10分类和40分类

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')  # 获取点云的路径
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        # 将分类的名称从txt文件中读出来放入列表，例如 下面读的就是10分类的结果
        # ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # 将类别名字和索引对应起来放入自动  例如 0<--->airplane

        shape_ids = {}
        if self.num_category == 10:  # 将待训练的点云的名字从之前划分好的txt文件中读取出来。即将训练 测试 集中点云的名字拿出来 放入shae id字典 如 batchub_001
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in
                       shape_ids[split]]  # 去除shape_ids中的下划线，拿到图像的名称存入列表  如  batchub_001--->bathtub
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        # 遍历之前的shapeids ， 将路径进行组合，找到对于图像对应的路径，并生成一个元组,将其存入self.datapath。其中，第一个元素是它的名称，第二个元素是它对于的路径。详情如下
        # ('bathtub', 'D:\\1Apython\\Pycharm_pojie\\3d\\Pointnet_Pointnet2_pytorch-master\\data\\modelnet40_normal_resampled\\bathtub\\bathtub_0001.txt')
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            "第一次运行会处理一些数据，将txt文件转换为.dat 文件 存入 self.save中"
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            # self.process_data就第一次执行的时候会 为True ,大多数情况都是执行下面的代码
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]  # 从self.datapath中获取点云数据，格式为（‘类别名称’,路径）
            cls = self.classes[self.datapath[index][0]]  # 根据索引和类别名称的对应关系 将名称与索引对应起来 例如  airplane<---->0
            label = np.array([cls]).astype(np.int32)  # 转换为数组形式
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)  # 使用np读点云数据 点云数据结果为 10000x6
            """
            读取txt文件我们通常使用 numpy 中的 loadtxt（）函数

            numpy.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0）

            注：loadtxt的功能是读入数据文件，这里的数据文件要求每一行数据的格式相同。 delimiter：数据之间的分隔符。如使用逗号","。
            """
            if self.uniform:  # 默认为False,没有使用FPS算法进行筛选，降采样到self.npoints个点
                point_set = farthest_point_sample(point_set, self.npoints)
            else:  # 取前1024个数
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 只对点的前三个维度进行归一化，即坐标的归一化
        if not self.use_normals:  # 如果不使用rgb信息 ，则 返只返回点云的xyz信息
            point_set = point_set[:, 0:3]

        return point_set, label[0]  # 返回读取到的 经过降采样的点云数据 和标签

    def __getitem__(self, index):
        return self._get_item(index)  # 调用self._get_item获取点云数据


def load_color_partseg():
    colors = []
    labels = []
    f = open("misc/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("misc/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()
        # self.partseg_colors = load_color_partseg()

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class ModelNet40(data.Dataset):
    def __init__(self, root, npoints=1024, split='train', normalize=True, data_augmentation=False):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.cat2id = {}

        # load classname and class id
        with open('misc/modelnet40category2id.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = int(ls[1])
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        # find all .h5 files
        with open(os.path.join(self.root, '{}_files.txt'.format(split)), 'r') as f:
            data_files = [os.path.join(self.root, line.strip().split('/')[-1]) for line in f]

        # load data from .h5 files
        point_clouds, labels = [], []
        for filename in data_files:
            with h5py.File(filename, 'r') as data_file:
                point_clouds.append(np.array(data_file['data']))
                labels.append(np.array(data_file['label']))
        self.pcs = np.concatenate(point_clouds, axis=0)
        self.lbs = np.concatenate(labels, axis=0)

    def __getitem__(self, index):
        point_cloud = self.pcs[index]
        label = self.lbs[index]
        classname = self.id2cat[label[0]]

        # select self.npoints from the original point cloud randomly
        choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        point_cloud = point_cloud[choice, :]

        # normalize into a sphere whose radius is 1
        if self.normalize:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            dist = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
            point_cloud = point_cloud / dist

        # data augmentation - random rotation and random jitter
        if self.data_augmentation:
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            # point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter
            r11, r12 = np.random.randint(1, 5, 2)
            # point_set_11 = random_augmentation(point_cloud, r11)
            # point_set_12 = random_augmentation(point_set_11, r12)

            r21, r22 = np.random.randint(1, 5, 2)
            point_set_21 = random_augmentation(point_cloud)
            # point_set_22 = random_augmentation(point_set_21, r22)
            return point_cloud, point_set_21

        if not self.data_augmentation:
            point_cloud = trans_3(point_cloud)
            label = torch.from_numpy(label)
            return point_cloud, label, classname

    def __len__(self):
        return len(self.pcs)


def load_ScanObjectNN(partition):
    BASE_DIR = '/home/lx/桌面/h5_files'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""


class NeighborsDataset(data.Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        # transform = dataset.transform

        # if isinstance(transform, dict):
        #     self.anchor_transform = transform['standard']
        #     self.neighbor_transform = transform['augment']
        # else:
        #     self.anchor_transform = transform
        #     self.neighbor_transform = transform

        # dataset.transform = None
        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors + 1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor, target, _ = self.dataset.__getitem__(index)
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor, _, _ = self.dataset.__getitem__(neighbor_index)

        # anchor['image'] = self.anchor_transform(anchor['image'])
        # neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor
        output['neighbor'] = neighbor
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = target

        return output


def visualization(point_set):
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(point_set[:, 0], point_set[:, 1], point_set[:, 2], c='y')
    # plt.show()
    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_set)
    open3d.visualization.draw_geometries([pcd])


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def random_augmentation(point_set):
    # if r == 1:
    theta = np.random.uniform(0, np.pi * 2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
    # elif r == 2:
    point_set = point_set * np.random.uniform(0.8, 1.25, 1)  # random zoom in/out
    # elif r == 3:
    point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
    # elif r == 4:
    point_set[:, 0] = -point_set[:, 0]  # mirror
    # elif r == 5:
    shifts = np.random.uniform(-0.1, 0.1, (1, 3))
    point_set = point_set + shifts  # Randomly shift point cloud
    return point_set


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    """
    Data Source: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
    """

    def __init__(self, root, num_point=2048, split='train', use_normal=False):
        self.catfile = os.path.join(root, 'synsetoffset2category.txt')
        self.use_normal = use_normal
        self.num_point = num_point
        self.cache_size = 20000
        self.datapath = []
        self.root = root
        self.cache = {}
        self.meta = {}
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # self.cat -> {'class name': syn_id, ...}
        # self.meta -> {'class name': file list, ...}
        # self.classes -> {'class name': class id, ...}
        # self.datapath -> [('class name', single file) , ...]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        train_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'))
        test_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'))
        val_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'))

        for item in self.cat:
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            self.meta[item] = []

            if split is 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split is 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s [Option: ]. Exiting...' % split)
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35],
                            'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
                            'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Lamp': [24, 25, 26, 27],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Knife': [22, 23],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15]}

    @staticmethod
    def read_fns(path):
        with open(path, 'r') as file:
            ids = set([str(d.split('/')[2]) for d in json.load(file)])
        return ids

    def __getitem__(self, index):
        if index in self.cache:
            pts, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat, pt = fn[0], np.loadtxt(fn[1]).astype(np.float32)
            cls = np.array([self.classes[cat]]).astype(np.int32)
            pts = pt[:, :6] if self.use_normal else pt[:, :3]
            seg = pt[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pts, cls, seg)

        choice = np.random.choice(len(seg), self.num_point, replace=True)
        pts[:, 0:3] = pc_normalize(pts[:, 0:3])
        pts, seg = pts[choice, :], seg[choice]

        return pts, cls, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    trainDataLoader = torch.utils.data.DataLoader(ModelNet40SVMPretrain(partition='train', num_points=2048), batch_size=16, shuffle=True)
    for i, data in enumerate(trainDataLoader):
        points1, points2 = data
        print(points1)



