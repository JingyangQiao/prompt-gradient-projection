import torch
import random
import numpy as np
from datasets.data_manager import DataManager
from utils import factory
from utils.toolkit import count_parameters


def run(args):
    seed = args["seed"]
    _set_random(seed)
    _set_device(args)
    train_and_evaluate(args)


def train_and_evaluate(args):
    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_class"], args["increment"], args)
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": []}
    for task in range(data_manager.nb_tasks):
        print("All params: {}".format(count_parameters(model._network)))
        print("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_acc = model.eval_task()
        model.after_task()
        print("CNN: {}".format(cnn_acc["grouped"]))
        cnn_curve["top1"].append(cnn_acc["top1"])
        print("CNN top1 curve: {}".format(cnn_curve['top1']))


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print("Seed Initialized!")
