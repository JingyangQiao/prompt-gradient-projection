import torch
import numpy as np


def get_representation_matrix(data_loader, device):
    count = 1
    representation = []
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        representation.append(input)
        count += 1
        if count > 768:
            representation = torch.cat(representation)
            break
    return representation


def get_rep(model, original_model, mem_example, task_id):
    rep = []
    rep_key = []
    for bs_ in range(32):
        # Prompt Representation Matrix
        _ = model(mem_example[bs_ * 24:(bs_ + 1) * 24, ...], task_id=task_id, train=False)
        rep_ = model.act["rep"].reshape(-1, 196 * 768)
        rep.append(rep_)
        del _, rep_

        # Key Representation Matrix
        key_ = original_model(mem_example[bs_ * 24:(bs_ + 1) * 24, ...], train=False)
        rep_key_ = key_["pre_logits"]
        rep_key.append(rep_key_)
        del key_, rep_key_

    return rep, rep_key


def update_memory(representation, threshold, feature=None):
    representation = np.matmul(representation, representation.T)
    if feature is None:
        U, S, Vh = np.linalg.svd(representation, full_matrices=False)
        sval_total = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total
        r = np.sum(np.cumsum(sval_ratio) < threshold)
        feature = U[:, 0:r]
    else:
        U1, S1, Vh1 = np.linalg.svd(representation, full_matrices=False)
        sval_total = (S1 ** 2).sum()
        # Projected Representation
        act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
        U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
        # criteria
        sval_hat = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total
        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if r == 0:
            return feature
        # update GPM
        U = np.hstack((feature, U[:, 0:r]))
        if U.shape[1] > U.shape[0]:
            feature = U[:, 0:U.shape[0]]
        else:
            feature = U

    print('-'*40)
    print('Gradient Constraints Summary', feature.shape)
    print('-'*40)

    return feature
