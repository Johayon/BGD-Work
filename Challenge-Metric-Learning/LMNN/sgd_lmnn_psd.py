"""
Large-margin nearest neighbor metric learning.

with SGD and mini batch
"""

import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances
from base_metric import BaseMetricLearner
import scipy


# commonality between LMNN implementations
class _base_LMNN(BaseMetricLearner):
    def __init__(self, **kwargs):
        self.params = kwargs

    def transformer(self):
        return self.L


class SGD_LMCA(_base_LMNN):
    """
    LMCA Learns a metric using large-margin nearest neighbor metric learning.
    LMCA(X, labels, k, d).fit()
    Learn a metric on X (NxD matrix) and labels (Nx1 vector).
    k: number of neighbors to consider, (does not include self-edges)
    d: dimension reduction size of L dxD matrix 
    regularization: weighting of pull and push terms

    """
    def __init__(self, d=50, min_iter=10, max_iter=100, psd_step=10, learn_rate=1e-3,
               regularization=0.5, convergence_tol=0.001, sgd_tuples=1000, mini_batch=10):
        _base_LMNN.__init__(self, d=d,min_iter=min_iter, max_iter=max_iter, psd_step=psd_step,
                        learn_rate=learn_rate, regularization=regularization,
                        convergence_tol=convergence_tol, sgd_tuples=sgd_tuples,mini_batch=mini_batch)
    
    def _process_inputs(self, X, labels, L):
        num_pts = X.shape[0]
        assert len(labels) == num_pts
        bb = np.array([ len(labels[labels == l])>1 for l in labels])
        self.labels2 = np.unique(labels[bb])
        self.allLabels = labels
        self.unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        self.labels = np.arange(len(self.unique_labels))
        self.X = X
        d = self.params['d']
        if L == []:
            self.L  = np.c_[np.eye(d),np.zeros((d,X.shape[1]-d))]
        else:
            self.L = L.copy()
        assert self.L.shape == (d,X.shape[1]), ('L is not well defined must be of the form (%d' % d + ',%d)' % X.shape[1])
        assert d <= X.shape[1], ('d cannot be of higher dimension than the data')
        self.sgd_obj = self.params['sgd_tuples']
        self.idx_obj = self._create_tuples(self.sgd_obj)

    def fit(self, X, labels, L=[], verbose=False):
        d = self.params['d']
        reg = self.params['regularization']
        learn_rate = self.params['learn_rate']
        psd_step = self.params['psd_step']
        convergence_tol = self.params['convergence_tol']
        min_iter = self.params['min_iter']
        mini_batch = self.params['mini_batch']
        if verbose:
            print "processing inputs"
        self._process_inputs(X, labels, L)
        if verbose:
            print "processing ended"
        L = self.L

        objective = self._compute_obj(self.idx_obj,L)
        print "First objective", objective
        # main loop
        G = np.zeros((X.shape[1],X.shape[1]))
        
        for it in xrange(1, self.params['max_iter']):

            # get tuple for descent
            for i in range(mini_batch):
                idx_tuple = self._create_tuples_hard(1,L)[0]

                # do the gradient update
                G += self._compute_gradient(idx_tuple,L)

            objective_old = objective
            # compute the objective function
            L_old = L
            L -= learn_rate * 2 * np.dot(L,G)
            G = np.zeros((X.shape[1],X.shape[1]))
            objective = self._compute_obj(self.idx_obj,L)
            assert not np.isnan(objective)
            delta_obj = objective - objective_old

            if verbose:
                print it, objective, delta_obj, learn_rate

            # update step size
            if delta_obj > 0 and it%psd_step != 1:
                # we're getting worse... roll back!
                # L = L_old
                learn_rate /= 2.0
                # objective = objective_old
            else:
                # update learn_rate
                learn_rate *= 1.01
                
            if it%psd_step == 0:
                print 'recalculate the projection to find the global'
                M = np.dot(L.T,L)
                M = self._psd_proj(M)
                ML = np.linalg.cholesky(M + 1e-10 * np.eye(M.shape[0]))
                L = ML.T[:d,:]
                
            # check for convergence
            if it > min_iter and abs(delta_obj) < convergence_tol and it%psd_step != 1:
                if verbose:
                    print "LMCA converged with objective", objective
                break
            else:
                if verbose:
                    print "LMCA didn't converge in %d steps." % it

        # store the last L
        print 'Final projection'
        M = np.dot(L.T,L)
        M = self._psd_proj(M)
        ML = np.linalg.cholesky(M + 1e-10 * np.eye(M.shape[0]))
        self.L = ML.T[:d,:]
        print "Last objective", objective
        return self

    def metric(self):
        return self.L.T.dot(self.L)

    def transform(self, X=None):
        if X is None:
            X = self.X
        return np.dot(X,self.L.T)

    def _create_tuples(self,n):
        m = self.labels2.shape[0]
        firstLabels = np.random.randint(0,m,n)
        indexes = np.arange(self.X.shape[0])
        tuples = []
        for i in range(n):
            idx_labels = indexes[self.allLabels == self.labels2[firstLabels[i]]]
            idx =  np.random.randint(0,len(idx_labels),2)
            idx_nolabels = indexes[self.allLabels != self.labels2[firstLabels[i]]]
            idx2 = np.random.randint(0,len(idx_nolabels),1)
            tuples.append([idx_labels[idx[0]],idx_labels[idx[1]],idx_nolabels[idx2][0]])
        return tuples

    def _create_tuples_hard(self,n,L):
        m = self.labels2.shape[0]
        p = self.unique_labels.shape[0]
        firstLabels = np.random.randint(0,m,n)
        X = self.X
        indexes = np.arange(self.X.shape[0])
        tuples = []
        for i in range(n):
            idx_labels = indexes[self.allLabels == self.labels2[firstLabels[i]]]
            idx_1 = idx_labels[np.random.randint(0,len(idx_labels),1)[0]]
            r = idx_labels[idx_labels != idx_1]
            XA = np.dot(np.reshape(X[idx_1],(1,1500)),L.T)
            XB = np.dot(np.reshape(X[r],(len(r),1500)),L.T)
            Diff = XA - XB
            arg = np.argmax(np.linalg.norm(Diff,axis=1))
            idx_2 = r[arg]
            j = np.random.randint(0,p-1,1)
            secondLabel = self.unique_labels[self.unique_labels != self.labels2[firstLabels[i]]][j]
            idx_nolabels = indexes[self.allLabels == secondLabel]
            XB = np.dot(np.reshape(X[idx_nolabels],(len(idx_nolabels),1500)),L.T)
            Diff = XA - XB
            arg = np.argmin(np.linalg.norm(Diff,axis=1))
            idx_3 = idx_nolabels[arg]
            tuples.append([idx_1,idx_2,idx_3])
        return tuples

    def _compute_obj(self, tuples, L):
        obj = 0
        X = self.X
        reg = self.params['regularization']
        for tup in tuples:
            obj += (1.0-reg)*np.sum(np.dot(X[tup[0]]-X[tup[1]],L.T)**2)
            obj += reg*( np.max([0,1 + np.sum(np.dot(X[tup[0]]-X[tup[1]],L.T)**2) - np.sum(np.dot(X[tup[0]]-X[tup[2]],L.T)**2)]))
        return obj

    def _compute_gradient(self, tup, L):
        X = self.X
        reg = self.params['regularization']
        diff = X[tup[0]]-X[tup[1]]
        gradient = (1.0-reg)*np.outer(diff, diff)
        diff2 = X[tup[0]]-X[tup[2]]
        if 1 + np.sum(np.dot(diff,L.T)**2) - np.sum(np.dot(diff2,L.T)**2) > 0:
            gradient += reg*(np.outer(diff, diff) - np.outer(diff2, diff2))
        return gradient
    
    def _psd_proj(self,M):
        """ projection de la matrice M sur le cone des matrices semi-definies positives"""
        # calcule des valeurs et vecteurs propres
        eigenval, eigenvec = np.linalg.eigh(M)
        # on trouve les valeurs propres negatives ou tres proches de 0
        ind_pos = eigenval > 1e-10
        # on reconstruit la matrice en ignorant ces dernieres
        M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
                   eigenvec[:, ind_pos].T)
        return M