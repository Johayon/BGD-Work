import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import metrics

plt.close('all')

############################################################################
#            Loading and visualizing the data
############################################################################

# if 1:  # use iris
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target
# else:  # use digits
#     digits = datasets.load_digits()
#     X = digits.data
#     y = digits.target

#     # on ne garde que les 3 premieres classes par simplicite
#     X = X[y < 3]
#     y = y[y < 3]

# # standardize data
# X -= X.mean(axis=0)
# X /= X.std(axis=0)
# X[np.isnan(X)] = 0.


def plot_2d(X, y):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""
    plt.figure()
    symlist = ['o', 's', '*', 'x', 'D', '+', 'p', 'v', 'H', '^']
    collist = ['blue', 'red', 'purple', 'orange', 'salmon', 'black', 'grey',
               'fuchsia']

    labs = np.unique(y)
    idxbyclass = [y == labs[i] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.plot(X[idxbyclass[i], 0], X[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(X[:, 1]), np.max(X[:, 1])])
    plt.xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    plt.show()

############################################################################
#            Displaying labeled data
############################################################################

# on utilise PCA pour projeter les donnees en 2D
# pca = decomposition.PCA(n_components=2)
# X_2D = pca.fit_transform(X)
# plot_2d(X_2D, y)

############################################################################
#                Stochastic gradient for metric learning
############################################################################


def psd_proj(M):
    """ projection de la matrice M sur le cone des matrices semi-definies
    positives"""
    # calcule des valeurs et vecteurs propres
    eigenval, eigenvec = np.linalg.eigh(M)
    # on trouve les valeurs propres negatives ou tres proches de 0
    ind_pos = eigenval > 1e-10
    # on reconstruit la matrice en ignorant ces dernieres
    M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
               eigenvec[:, ind_pos].T)
    return M


def hinge_loss_pairs(X, pairs_idx, y_pairs, M):
    """Calcul du hinge loss sur les paires
    """
    diff = X[pairs_idx[:, 0], :] - X[pairs_idx[:, 1], :]
    return np.maximum(0.0, 1.0 + y_pairs.T * (np.sum(
                                 np.dot(M, diff.T) * diff.T, axis=0) - 2.0))


# def sgd_metric_learning(X, y, gamma, n_iter, n_eval, M_ini, random_state=42, pairs_idx=[]):
#     """Stochastic gradient algorithm for metric learning
    
#     Parameters
#     ----------
#     X : array, shape (n_samples, n_features)
#         The data
#     y : array, shape (n_samples,)
#         The targets.
#     gamma : float | callable
#         The step size. Can be a constant float or a function
#         that allows to have a variable step size
#     n_iter : int
#         The number of iterations
#     n_eval : int
#         The number of pairs to evaluate the objective function
#     M_ini : array, shape (n_features,n_features)
#         The initial value of M
#     random_state : int
#         Random seed to make the algorithm deterministic
#     """
#     rng = np.random.RandomState(random_state)
#     n_samples = X.shape[0]
#     n_features = X.shape[1]
    
#     # pour eviter d'evaluer le risque sur toutes les paires possibles
#     # on tire n_eval paires aleatoirement
#     pairs_idx, y_pairs = generate_pairs(y, n_eval, 0.1)
#     # calcul du label des paires
#     M = M_ini.copy()
#     pobj = np.zeros(n_iter)
    
#     if not callable(gamma):
#         # Turn gamma to a function
#         gamma_func = lambda t: gamma
#     else:
#         gamma_func = gamma

#     for t in range(n_iter):
#         pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx, y_pairs, M))
#         gradient = np.zeros((n_features,n_features))
#         idx = rng.randint(0, n_eval)
#         if hinge_loss_pairs(X, pairs_idx[idx:idx+1], y_pairs[idx:idx+1], M)[0] != 0.:
#             Xt = np.reshape(X[pairs_idx[idx,0],:] - X[pairs_idx[idx,1],:],(n_features,1))
#             gradient = y_pairs[idx]*np.dot(Xt,Xt.T)
#         M -= gamma_func(t) * gradient
#         M = psd_proj(M)
#     return M, pobj

def sgd_metric_learning(X, y, gamma, n_iter, n_eval, M_ini, random_state=42):
    """Stochastic gradient algorithm for metric learning
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets.
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_iter : int
        The number of iterations
    n_eval : int
        The number of pairs to evaluate the objective function
    M_ini : array, shape (n_features,n_features)
        The initial value of M
    random_state : int
        Random seed to make the algorithm deterministic
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    # tirer n_eval paires aleatoirement
    pairs_idx, y_pairs = generate_pairs(y, n_eval, 0.1)
    pairs_idx2, y_pairs2 = generate_pairs(y, n_eval, 0.1)
    # calcul du label des paires
    y_pairs = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
    y_pairs2 = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
    M = M_ini.copy()
    pobj = np.zeros(n_iter)
    
    if not callable(gamma):
        # Turn gamma to a function for QUESTION 5
        gamma_func = lambda t: gamma
    else:
        gamma_func = gamma

    for t in range(n_iter):
        pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx, y_pairs, M))
        idx = rng.randint(0, n_eval)
        diff = X[pairs_idx2[idx,0], :] - X[pairs_idx2[idx,1], :]
        y_idx = y_pairs2[idx]
        gradient = (y_idx * np.outer(diff, diff) *
                    ((1.0 + y_idx * (np.dot(diff, np.dot(M, diff.T)) - 2.0)) > 0))
        M -= gamma_func(t) * gradient
        if t%1000 == 0:
            M = psd_proj(M)
    return M, pobj




# n_features = X.shape[1]

# M_ini = np.eye(n_features)
# M, pobj = sgd_metric_learning(X, y, 0.001, 10000, 1000, M_ini)

# plt.figure()
# plt.plot(pobj)
# plt.xlabel('t')
# plt.ylabel('cost')
# plt.title('hinge stochastic')
# plt.show()

# calcul de la factorisation de cholesky
# on ajoute de tres faibles coefficients sur la diagonale pour eviter
# les erreurs numeriques
#L = np.linalg.cholesky(M + 1e-10 * np.eye(n_features))
# on projette lineairement les donnees
#X_proj = np.dot(X, L)

# pca = decomposition.PCA(n_components=2)
# X_2D = pca.fit_transform(X_proj)
# plot_2d(X_2D, y)

# TODO QUESTION 5: tirer paires aleatoires
# calculer les distances et tracer les courbes ROC

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
    for i in range(n_pairs):
        if rand_vec[i] <= positive_ratio:
            idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_same.shape[0])
            pairs_idx[i, 1] = idx_same[idx2]
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

# pairs_idx, pairs_label = generate_pairs(y, 1000, 0.1)

# dist_euc = euc_dist_pairs(X, pairs_idx)

# dist_M = euc_dist_pairs(X_proj, pairs_idx)

# fpr_e, tpr_e, thresholds_e = metrics.roc_curve(pairs_label, -dist_euc)
# fpr, tpr, thresholds = metrics.roc_curve(pairs_label, -dist_M)

# plt.clf()
# plt.plot(fpr, tpr, label='ROC curve')
# plt.plot(fpr_e, tpr_e, label='ROC curve')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()