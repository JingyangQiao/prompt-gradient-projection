import torch
import numpy as np
from sklearn.decomposition import PCA


def get_prefix_matrix(data_loader, model, device):
    model.eval()
    count = 1
    representation_g = {}
    representation_e = {}
    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            # prefix = torch.randn(1, 2, 5, 12, 64)
            _ = model(input)
            del _

            for layer in model.g_prefix_feature:
                if layer not in representation_g:
                    representation_g[layer] = {"key": []}
                representation_g[layer]["key"].append(model.g_prefix_feature[layer]["key"])
            for layer in model.e_prefix_feature:
                if layer not in representation_e:
                    representation_e[layer] = {"key": []}
                representation_e[layer]["key"].append(model.e_prefix_feature[layer]["key"])
            count += 1

            if count > 768:
                for layer in representation_g:
                    for item in representation_g[layer]:
                        representation_g[layer][item] = torch.cat(representation_g[layer][item])
                        representation_g[layer][item] = representation_g[layer][item].detach().cpu().numpy()
                        representation_g[layer][item] = representation_g[layer][item].reshape(representation_g[layer][item].shape[0], -1)
                        rep = representation_g[layer][item]
                        pca = PCA(n_components=50)
                        pca = pca.fit(rep)
                        rep = pca.transform(rep)
                        representation_g[layer][item] = rep

                for layer in representation_e:
                    for item in representation_e[layer]:
                        representation_e[layer][item] = torch.cat(representation_e[layer][item])
                        representation_e[layer][item] = representation_e[layer][item].detach().cpu().numpy()
                        representation_e[layer][item] = representation_e[layer][item].reshape(representation_e[layer][item].shape[0], -1)
                        rep = representation_e[layer][item]
                        pca = PCA(n_components=50)
                        pca = pca.fit(rep)
                        rep = pca.transform(rep)
                        representation_e[layer][item] = rep

                break
    torch.cuda.empty_cache()

    return representation_g, representation_e


def update_memory_prefix(represent, threshold, features=None):
    for layer in represent:
        for item in represent[layer]:
            representation = represent[layer][item]
            representation = np.matmul(representation, representation.T)
            try:
                feature = features[layer][item]
            except:
                feature = None
            if feature is None:
                if layer not in features:
                    features[layer] = {}
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
                    feature = feature
                # update GPM
                U = np.hstack((feature, U[:, 0:r]))
                if U.shape[1] > U.shape[0]:
                    feature = U[:, 0:U.shape[0]]
                else:
                    feature = U
            print('-'*40)
            print('Gradient Constraints Summary', feature.shape)
            print('-'*40)
            features[layer][item] = feature

    return features
