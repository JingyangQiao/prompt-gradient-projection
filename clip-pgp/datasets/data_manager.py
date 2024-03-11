import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import pickle
import os

from datasets.data import iCifar100


class DataManager():
    def __init__(self, dataset_name, shuffle, seed, init_class, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        self._increments = [init_class]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, mode, appendent=None, ret_data=False):
        if mode == "train":
            trans = transforms.Compose([*self._train_trans, *self._common_trans])
        elif mode == "flip":
            trans = transforms.Compose([*self._test_trans, transforms.RandomHorizontalFlip(p=1.), *self._common_trans])
        elif mode == "test":
            trans = transforms.Compose([*self._test_trans, *self._common_trans])
        else:
            raise ValueError("Unknown mode {}".format(mode))

        if mode == "train" and self.dataset_name == "cifar100":
            dataset = datasets.CIFAR100(self.args["data_path"], train=True, download=True)
        elif mode == "test" and self.dataset_name == "cifar100":
            dataset = datasets.CIFAR100(self.args["data_path"], train=False, download=True)
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(dataset, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, transforms=trans)
        else:
            return DummyDataset(data, targets, transforms=trans)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        self._train_trans = idata.train_trans
        self._test_trans = idata.test_trans
        self._common_trans = idata.common_trans
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(idata.class_order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        print("class_order", self._class_order)

    def _select(self, dataset, low_range, high_range):
        x = []
        y = []
        x_data = dataset.data
        y_data = np.array(dataset.targets)
        for idx in range(len(x_data)):
            image = x_data[idx]
            target = y_data[idx]
            if high_range > int(target) >= low_range:
                x.append(Image.fromarray(image))
                y.append(target.astype("uint8"))
        return np.array(x), np.array(y)


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar100":
        return iCifar100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DummyDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trans = transforms
        self.tasks = labels // 10

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.trans:
            self.images[item] = self.trans(self.images[item])
        return self.tasks[item], self.images[item], self.labels[item]


#######################################################Test#############################################################
# args = {"data_path": "/data/qiaojingyang/cifar100"}
# manager = DataManager("cifar100", shuffle="False", seed=0, init_class=10, increment=10, args=args)
# indices = [i for i in range(10, 20)]
# set = manager.get_dataset(indices=indices, mode="train", appendent=None, ret_data=False)
# print(set[0][1].size())
# set = manager.get_dataset(indices=indices, mode="test", appendent=None, ret_data=False)
# print(set[0][1].size())