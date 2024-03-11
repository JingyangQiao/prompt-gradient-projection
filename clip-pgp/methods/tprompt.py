import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

from methods.base import BaseLearner
from models.tclip import Ticlip
from utils.toolkit import tensor2numpy
from utils import memory


class TPrompts(BaseLearner):
    def __init__(self, args):
        super(TPrompts, self).__init__(args)
        self._network = Ticlip(args)
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.lr_decay = args["lr_decay"]
        self.weight_decay = args["weight_decay"]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.topk = 2
        self.class_num = self._network.class_num  # 10
        self.all_keys = []
        self.feature = None
        self.feature_mat = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc()
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), mode="test")
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "text_prompt_pool" + "." + str(self._network.task - 1) in name:
                param.requires_grad_(True)
            # if "image_prompt_pool" + "." + str(self._network.task - 1) in name:
            #     param.requires_grad_(True)
            if "image_prompt_pool" in name:
                param.requires_grad_(True)

        trainable = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                trainable.add(name)
        print(f"Parameters to be updated: {trainable}")

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epochs)
            self.run_epoch = self.init_epochs
            self.train_function(train_loader, test_loader, optimizer, schedule)

        else:
            print("prompt feature shape", self.feature.shape)
            self.feature_mat = torch.Tensor(np.dot(self.feature, self.feature.transpose())).to(self._device)
            print('Prompt Projection Matrix Shape: {}'.format(self.feature_mat.shape))
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lr, weight_decay=self.weight_decay)
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, schedule, feature_mat=self.feature_mat)

        # Image Gradient Projection Matrix
        self._network.eval()
        mem_example, mem_tasks = memory.get_representation_matrix(train_loader, self._device)
        rep = []
        with torch.no_grad():
            for bs_ in range(24):
                rep_ = self._network.query(mem_example[bs_ * 32:(bs_ + 1) * 32, ...], mem_tasks[bs_ * 32:(bs_ + 1) * 32])
                rep_ = rep_.reshape(32, -1)
                rep.append(rep_)
        rep = torch.cat(rep)
        rep = rep.detach().cpu().numpy()
        pca = PCA(n_components=5)
        pca = pca.fit(rep)
        rep = pca.transform(rep)
        self._network.train()

        # Prompt Gradient Projection Matrix
        for k, (m, params) in enumerate(self._network.named_parameters()):
            if "image_prompt_pool" in m:
                p_ = params.data
                p_ = p_.view(-1, 768).detach().cpu().numpy().transpose(1, 0)
                pca = PCA(n_components=5)
                pca = pca.fit(p_)
                p_ = pca.transform(p_)

        # pca = PCA(n_components=9)
        # pca = pca.fit(p_)
        # p = pca.transform(p_)
        rep = rep + p_
        self.feature = memory.update_memory(rep, 0.90, self.feature)

    def train_function(self, train_loader, test_loader, optimizer, schedule, feature_mat=None):
        bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (tasks, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), torch.tensor(targets, dtype=torch.long).to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)
                logits = self._network(inputs)
                loss = F.cross_entropy(logits, targets % 10)
                optimizer.zero_grad()
                loss.backward()

                # Gradient Projection Step
                if feature_mat is not None:
                    for k, (m, params) in enumerate(self._network.named_parameters()):
                        if "image_prompt_pool" in m:
                            params.grad.data = params.grad.data - torch.matmul(params.grad.data, feature_mat)

                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq((targets % 10).expand_as(preds)).cpu().sum()
                total += len(targets)
            schedule.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train Acc {:.2f}, Test Acc {:.2f}".format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc
            )
            bar.set_description(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        print("Eval Task Start.")
        for _, (tasks, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            tasks = tasks.to(self._device)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, tasks)
                else:
                    outputs = self._network.interface(inputs, tasks)
            if self.args["mode"] == "TIL":
                predicts = torch.max(outputs, dim=1)[1] + tasks * 10
            elif self.args["mode"] == "CIL":
                predicts = torch.max(outputs, dim=1)[1]
            else:
                raise NotImplementedError
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets % 10).sum()
            total += len(inputs)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
