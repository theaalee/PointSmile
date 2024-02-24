from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from data import ModelNet40SVM, ShapeNetDatasetCpt,ModelNetDataLoader
from model import PointNetfeat, DGCNN
import torch
import numpy as np


train_val_loader = DataLoader(ModelNetDataLoader(root='/home/lx/桌面/modelnet40_normal_resampled', num_category=10, split='train'), batch_size=64)
test_val_loader = DataLoader(ModelNetDataLoader(root='/home/lx/桌面/modelnet40_normal_resampled', num_category=10, split='test'), batch_size=64)

feats_train = []
labels_train = []
feats_test = []
labels_test = []
point_model = PointNetfeat().cuda(0)
point_model.load_state_dict(torch.load('/home/lx/桌面/best_pn_model.pth'))

for i, (data, label) in enumerate(train_val_loader):
    data = data.permute(0, 2, 1).to(0)
    with torch.no_grad():
        feats, _, _, _ = point_model(data)
    feats = feats.detach().cpu().numpy()
    for feat in feats:
        feats_train.append(feat)
    labels_train += label

feats_train = np.array(feats_train)

for i, (data, label) in enumerate(test_val_loader):
    data = data.permute(0, 2, 1).to(0)
    with torch.no_grad():
        feats, _, _, _ = point_model(data)
    feats = feats.detach().cpu().numpy()
    for feat in feats:
        feats_test.append(feat)
    labels_test += label

feats_train = np.array(feats_train)
labels_train = np.array(labels_train)
feats_test = np.array(feats_test)
labels_test = np.array(labels_test)

digits = feats_train

plt.subplot(121)
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(digits)
# X_2d_min, X_2d_max = X_2d.min(0), X_2d.max(0)
# X_2d = (X_2d-X_2d_min)/(X_2d_max-X_2d_min)
y = labels_train
target_ids = range(len(y))

_, ax = plt.subplots(figsize=(10, 5))

# plt.figure(figsize=(10, 5))
colors = ['red', 'blue', 'navy', 'green', 'violet', 'brown', 'gold', 'lime', 'teal', 'olive']
# ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

# c=np.random.rand(1,3)
for i in target_ids:
    if i<10:
        ax.scatter(X_2d[y == i, 0], X_2d[y == i, 1], s=10, c=colors[i])
# plt.legend()
plt.show()
