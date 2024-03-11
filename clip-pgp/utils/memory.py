import torch
import numpy as np


def get_representation_matrix(data_loader, device):
    count = 1
    representation, rep_tasks = [], []
    for tasks, inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        representation.append(inputs)
        rep_tasks.append(tasks)
        count += 1
        if count > 24:
            representation = torch.cat(representation)
            rep_tasks = torch.cat(rep_tasks)
            break
    return representation, rep_tasks


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
