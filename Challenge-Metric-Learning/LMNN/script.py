import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import io
import bisect

train_facile = io.loadmat('traind.mat')

def generate_pairs(label, n_pairs, positive_ratio, random_state=42):
    """Generate a set of pair indices
    
    Parameters
    ----------
    label : array, shape (n_samples, 1)
        Label vector
    n_pairs : int
        Number of pairs to generate
    positive_ratio : float
        Positive to negative ratio for pairs
    random_state : int
        Random seed for reproducibility
        
    Output
    ------
    pairs_idx : array, shape (n_pairs, 2)
        The indices for the set of pairs
    label_pairs : array, shape (n_pairs, 1)
        The pair labels (+1 or -1)
    """
    rng = np.random.RandomState(random_state)
    n_samples = label.shape[0]
    pairs_idx = np.zeros((n_pairs, 2), dtype=int)
    pairs_idx[:, 0] = rng.randint(0, n_samples, n_pairs)
    rand_vec = rng.rand(n_pairs)
    k=0
    k_max = positive_ratio * n_pairs + 2
    for i in range(n_pairs):
        if rand_vec[i] <= 0.5 and len(np.where(label == label[pairs_idx[i, 0]])[0]) > 1 and k < k_max:
            idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_same.shape[0])
            while idx_same[idx2] == pairs_idx[i, 0]:
                idx2 = rng.randint(idx_same.shape[0])
            pairs_idx[i, 1] = idx_same[idx2]
            k+=1
        else:
            idx_diff = np.where(label != label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_diff.shape[0])
            pairs_idx[i, 1] = idx_diff[idx2]
    pairs_label = 2.0 * (label[pairs_idx[:, 0]] == label[pairs_idx[:, 1]]) - 1.0
    return pairs_idx, pairs_label

def euc_dist_pairs(X, pairs, batch_size=10000):
    """Compute an array of Euclidean distances between points indexed by pairs

    To make it memory-efficient, we compute the array in several batches.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data matrix
    pairs : array, shape (n_pairs, 2)
        Pair indices
    batch_size : int
        Batch size (the smaller, the slower but less memory intensive)
        
    Output
    ------
    dist : array, shape (n_pairs,)
        The array of distances
    """
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    for a in range(0, n_pairs, batch_size):
        b = min(a + batch_size, n_pairs)
        dist[a:b] = np.sqrt(np.sum((X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]) ** 2, axis=1))
    return dist


from sklearn import decomposition

X = train_facile['X']
y = np.array(train_facile['label']).flatten()

import imp
import sgd_lmnn_psd
imp.reload(sgd_lmnn_psd)
from sgd_lmnn_psd import SGD_LMCA

L2 = np.loadtxt('Matrice_save4.txt')

sgd_lmca = SGD_LMCA(d=1100,max_iter=200,psd_step=20,mini_batch=2000,sgd_tuples=4000,learn_rate=0.0001,regularization=0.75,convergence_tol=0.005)

sgd_lmca.fit(X, y,verbose=True, L=L2)

L3 = sgd_lmca.L

np.savetxt('Matrice_save5.txt',L3)


