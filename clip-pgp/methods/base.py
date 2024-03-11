import numpy as np
import torch
import torch.nn as nn

from utils.toolkit import accuracy


class BaseLearner:
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self.topk = 5

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T, y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.round((y_pred.T == np.tile(y_true,
                                                   (self.topk, 1))).sum() * 100 / len(y_true), decimals=2)
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.val_loader)
        cnn_acc = self._evaluate(y_pred, y_true)
        return cnn_acc

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _compute_accuracy(self, model, loader):
        pass

    def _eval_cnn(self, loader):
        pass
