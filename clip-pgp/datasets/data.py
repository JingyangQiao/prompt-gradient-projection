import numpy as np
from torchvision import transforms


class iData():
    train_trans = []
    test_trans = []
    common_trans = []
    class_order = None


class iCifar100(iData):
    scale = (0.05, 1.)
    ratio = (3./4., 4./3.)
    train_trans = [
        transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    test_trans = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    common_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]

    def __init__(self):
        class_order = np.arange(100).tolist()
        self.class_order = class_order
