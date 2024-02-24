import random
import numpy as np
import torch
import time


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist < dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids)
    return cids


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25, p=1):
        self.lo, self.hi = lo, hi
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudMirror(object):
    def __init__(self, p=1):
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points[:, 0] = -points[:, 0]
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0]), p=1):
        self.axis = axis
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        # points = torch.astensor(points)
        if self.axis is None:
            angles = np.random.uniform(size=3) * 2 * np.pi
            Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        else:
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = angle_axis(rotation_angle, self.axis)
        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t()).numpy()
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points.numpy()


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18, p=1):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.p = p

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05, p=1):
        self.std, self.clip = std, clip
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointCloudShift(object):
    def __init__(self, shift_range=0.1, p=1):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        shifts = np.random.uniform(-self.shift_range, self.shift_range, (3))
        points[:, 0:3] += shifts
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1, p=1):
        self.translate_range = translate_range
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        coord_min = np.min(points[:, :3], axis=0)
        coord_max = np.max(points[:, :3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        points[:, 0:3] += translation
        return torch.from_numpy(points).float()


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


# 0.95
class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875, p=1):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        pc = points.numpy()
        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


# Reorder points
class PointcloudSample(object):
    def __init__(self, num_pt=4096):
        self.num_points = num_pt

    def __call__(self, points):
        pc = points.numpy()
        # pt_idxs = np.arange(0, self.num_points)
        pt_idxs = np.arange(0, points.shape[0])
        np.random.shuffle(pt_idxs)
        pc = pc[pt_idxs[0:self.num_points], :]
        return torch.from_numpy(pc).float()


class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()


# remove (0,0,0)
class PointcloudRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


class PointcloudRandomCrop(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=4096, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(new_points).float()


class PointcloudRandomCutout(object):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()
        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1 - cut_ratio)
            new_coord_max = new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices == False]
                valid = True

        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        return torch.from_numpy(points).float()


class PointcloudUpSampling(object):
    def __init__(self, max_num_points, radius=0.1, nsample=5, centroid="random"):
        self.max_num_points = max_num_points
        # self.radius = radius
        self.centroid = centroid
        self.nsample = nsample

    def __call__(self, points):
        t0 = time.time()

        p_num = points.shape[0]
        if p_num > self.max_num_points:
            return points

        c_num = self.max_num_points - p_num

        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = fps(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)

        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)

        r_t = r.t()  # 转置
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)

        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]

        uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
        median = np.median(uniform, axis=1, keepdims=True)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
        choice = adj_topk[uniform > median]  # (c_num, n_samples)

        choice = choice.reshape(-1, self.nsample)

        sample_points = points[choice]  # (c_num, n_samples, 3)

        new_points = torch.mean(sample_points, dim=1)
        new_points = torch.cat([points, new_points], 0)

        return new_points


def check_aspect2D(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    return (xy_aspect >= aspect_min)


class PointcloudRandomCuboid(object):
    def __init__(self, p=1, crop=0.85, randcrop=1, aspect=0.75):
        self.crop = crop
        self.randcrop = randcrop
        self.aspect = aspect
        self.p = p

    def __call__(self, points):

        if np.random.uniform(0, 1) > self.p:
            return points

        points = points.numpy()
        range_xyz = np.max(points[:, 0:2], axis=0) - np.min(points[:, 0:2], axis=0)

        crop_range = self.crop + np.random.rand(2) * (self.randcrop - self.crop)

        loop_count = 0

        while not check_aspect2D(crop_range, float(self.aspect)):
            loop_count += 1
            crop_range = self.crop + np.random.rand(2) * (self.randcrop - self.crop)
            if loop_count > 100:
                break

        loop_count = 0

        while True:
            loop_count += 1

            sample_center = points[np.random.choice(len(points)), 0:3]

            new_range = range_xyz * crop_range / 2.0

            max_xyz = sample_center[0:2] + new_range
            min_xyz = sample_center[0:2] - new_range

            upper_idx = np.sum((points[:, 0:2] <= max_xyz).astype(np.int32), 1) == 2
            lower_idx = np.sum((points[:, 0:2] >= min_xyz).astype(np.int32), 1) == 2

            new_pointidx = (upper_idx) & (lower_idx)
            if (loop_count > 100) or (np.sum(new_pointidx) > float(10000)):
                break

        point_cloud = points[new_pointidx, :]

        return torch.tensor(point_cloud)


def points_sampler(points, num):
    pt_idxs = np.arange(0, points.shape[0])
    np.random.shuffle(pt_idxs)
    points = points[pt_idxs[0:num], :]
    return points


class PointcloudRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, points):
        if random.random() > self.p:
            points[:, 0] = -1 * points[:, 0]
        if random.random() > self.p:
            points[:, 1] = -1 * points[:, 1]
        return points


def random_shift_point_cloud(batch_data, shift_range=0.1):
    """ Shift the Point Cloud along the XYZ axis, magnitude is randomly sampled from [-0.1, 0.1] """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Scale the Point Cloud Objects into a Random Magnitude between [0.8, 1.25] """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


class PointcloudRandomDrop(object):
    def __init__(self, p=1, crop=0.2):
        self.crop = crop
        self.p = p

    def __call__(self, point_cloud):
        if random.random() > self.p:
            return point_cloud

        point_cloud = point_cloud.numpy()
        range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(point_cloud[:, 0:3], axis=0)

        crop_range = float(self.crop)
        new_range = range_xyz * crop_range / 2.0
        numb, numv = np.histogram(point_cloud[:, 2])
        max_idx = np.argmax(numb)
        minidx = max(0, max_idx - 2)
        maxidx = min(len(numv) - 1, max_idx + 2)
        range_v = [numv[minidx], numv[maxidx]]
        loop_count = 0
        # write_ply_color(point_cloud[:,:3], point_cloud[:,3:], "before.ply")
        while True:
            sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]
            loop_count += 1
            if (loop_count <= 100):
                if (sample_center[-1] > range_v[1]) or (sample_center[-1] < range_v[0]):
                    continue
            break
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = np.sum((point_cloud[:, 0:3] < max_xyz).astype(np.int32), 1) == 3
        lower_idx = np.sum((point_cloud[:, 0:3] > min_xyz).astype(np.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        point_cloud = point_cloud[new_pointidx, :]
        return point_cloud


if __name__ == '__main__':
    datapath = "/home/lx/桌面/shapenetcore_partanno_segmentation_benchmark_v0"
    import torchvision.transforms as transforms

    from data import ShapeNetDatasetCpt
    d = ShapeNetDatasetCpt(root=datapath, split='train', n_imgs=2500)
    ps, ps1 = d[2]
    import matplotlib.pylab as plt
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c='y')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(ps1[:, 0], ps1[:, 1], ps1[:, 2], c='y')
    plt.show()