"""
GB-Large-margin nearest neighbor metric learning.

with SGD and mini batch
"""

import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances
from base_metric import BaseMetricLearner
import scipy
from sklearn.tree import DecisionTreeRegressor

# commonality between LMNN implementations
class _base_LMNN(BaseMetricLearner):
    def __init__(self, **kwargs):
        self.params = kwargs

    def transformer(self):
        return self.L


class GB_SGD_LMCA(_base_LMNN):
    """
    GB_LMCA Learns trees using large-margin nearest neighbor metric learning.
    GB_LMCA(X, labels, k, d).fit()
    Learn a metric on X (NxD matrix) and labels (Nx1 vector).
    k: number of neighbors to consider, (does not include self-edges)
    d: dimension reduction size of L dxD matrix 
    regularization: weighting of pull and push terms

    """
    def __init__(self, d=50, min_iter=50, max_iter=100, learn_rate=1e-3,
               regularization=0.5, convergence_tol=0.001, sgd_tuples=1000, mini_batch=10,depth = 3):
        
        _base_LMNN.__init__(self, d=d,min_iter=min_iter, max_iter=max_iter,
                                  learn_rate=learn_rate, regularization=regularization,                  
                                  convergence_tol=convergence_tol,sgd_tuples=sgd_tuples,
                                  mini_batch=mini_batch,depth=depth)
    
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
        self.Trees = np.empty(shape=(self.params['max_iter'],d),dtype=object)
        if L == []:
            self.L  = np.c_[np.eye(d),np.zeros((d,X.shape[1]-d))]
        else:
            self.L = L.copy()
        assert self.L.shape == (d,X.shape[1]), ('L is not well defined must be of the form (%d' % d + ',%d)' % X.shape[1])
        assert d <= X.shape[1], ('d cannot be of higher dimension than the data')
        self.sgd_obj = self.params['sgd_tuples']
        self.idx_obj = self._create_tuples(self.sgd_obj)
        self.trees = np.empty(self.params['max_iter']).astype(object)

    def fit(self, X, labels, L=[], verbose=False):
        d = self.params['d']
        depth = self.params['depth']
        reg = self.params['regularization']
        learn_rate = self.params['learn_rate']
        convergence_tol = self.params['convergence_tol']
        min_iter = self.params['min_iter']
        mini_batch = self.params['mini_batch']
        if verbose:
            print "processing inputs"
        self._process_inputs(X, labels, L)
        if verbose:
            print "processing ended"
        L = self.L

        objective = self._compute_obj(self.idx_obj,L,0,learn_rate)
        print "First objective", objective
        # main loop
        G = []
        x_tree = []
            
        for it in range(self.params['max_iter']):

            # get tuple for descent
            for i in range(mini_batch):
                idx_tuple = self._create_tuples_hard(1,L,it)[0]
                x_tree.append(X[idx_tuple[0]])
                # do the gradient update
                G.append(2*np.sum(self._compute_gradient(idx_tuple,L,it,learn_rate),axis=1))

            objective_old = objective
            # compute the objective function
            g_t = G
            g_t = np.array(g_t)
            x_tree = np.array(x_tree)
            if verbose:
                print "training trees"
            self.trees[it] = DecisionTreeRegressor(max_depth=depth)
            y_tree = g_t
            self.trees[it].fit(x_tree,y_tree)
            if verbose:
                print "training trees ended"
            
            objective = self._compute_obj(self.idx_obj,L,it+1,learn_rate)
            x_tree = []
            G = []
            assert not np.isnan(objective)
            
            delta_obj = objective - objective_old

            if verbose:
                print it, objective, delta_obj, learn_rate

            # update step size
            if delta_obj > 0:
                # we're getting worse
                learn_rate /= 2.0
            else:
                # update learn_rate
                learn_rate *= 1.01

            # check for convergence
            if it > min_iter and abs(delta_obj) < convergence_tol:
                if verbose:
                    print "LMCA converged with objective", objective
                break
            else:
                if verbose:
                    print "LMCA didn't converge in %d steps." % it

        # store the last L
        self.L = L
        print "Last objective", objective
        return self

    def metric(self):
        return self.L.T.dot(self.L)

    def transform(self, X=None):
        if X is None:
            X = self.X
        return self.L.dot(X.T).T 

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

    def _create_tuples_hard(self,n,L,it):
        m = self.labels2.shape[0]
        p = self.unique_labels.shape[0]
        firstLabels = np.random.randint(0,m,n)
        X = self.X
        dim = X.shape[1]
        indexes = np.arange(self.X.shape[0])
        tuples = []
        for i in range(n):
            idx_labels = indexes[self.allLabels == self.labels2[firstLabels[i]]]
            idx_1 = idx_labels[np.random.randint(0,len(idx_labels),1)[0]]
            r = idx_labels[idx_labels != idx_1]
            XA = np.dot(np.reshape(X[idx_1],(1,dim)),L.T)
            XB = np.dot(np.reshape(X[r],(len(r),dim)),L.T)
            Diff = XA - XB
            arg = np.argmax(np.linalg.norm(Diff,axis=1))
            idx_2 = r[arg]
            j = np.random.randint(0,p-1,1)
            secondLabel = self.unique_labels[self.unique_labels != self.labels2[firstLabels[i]]][j]
            idx_nolabels = indexes[self.allLabels == secondLabel]
            XB = np.dot(np.reshape(X[idx_nolabels],(len(idx_nolabels),dim)),L.T)
            Diff = XA - XB
            arg = np.argmin(np.linalg.norm(Diff,axis=1))
            idx_3 = idx_nolabels[arg]
            tuples.append([idx_1,idx_2,idx_3])
        return tuples

    def _compute_obj(self, tuples,L,it,learn_rate):
        obj = 0
        X = self.X
        d = self.params['d']
        reg = self.params['regularization']
            
        for tup in tuples:
            X0_tree = np.zeros(d)
            X1_tree = np.zeros(d)
            X2_tree = np.zeros(d)
            for i in range(0,it):
                X0_tree += learn_rate*self.trees[i].predict(X[tup[0]])[0]
                X1_tree += learn_rate*self.trees[i].predict(X[tup[1]])[0]
                X2_tree += learn_rate*self.trees[i].predict(X[tup[2]])[0]
            
            obj += (1.0-reg)*np.sum((np.dot(L,X[tup[0]]-X[tup[1]]) + X0_tree -  X1_tree)**2)
            obj += reg*( np.max([0,1 + np.sum((np.dot(L,X[tup[0]]-X[tup[1]]) + X0_tree -  X1_tree)**2) - np.sum((np.dot(L,X[tup[0]]-X[tup[2]]) + X0_tree -  X1_tree)**2)]))
        return obj

    def _compute_gradient(self, tup, L,it,learn_rate):
        X = self.X
        reg = self.params['regularization']
        d = self.params['d']
        
        X0_tree = np.zeros(d)
        X1_tree = np.zeros(d)
        X2_tree = np.zeros(d)
        for i in range(0,it):
            X0_tree += learn_rate*self.trees[i].predict(X[tup[0]])[0]
            X1_tree += learn_rate*self.trees[i].predict(X[tup[1]])[0]
            X2_tree += learn_rate*self.trees[i].predict(X[tup[2]])[0]
        print X0_tree
        diff = X[tup[0]]-X[tup[1]]
        diff_tree = X0_tree - X1_tree
        diff_all = np.dot(diff,L.T)+diff_tree
        gradient = (1.0-reg)*np.outer(diff_all, diff_all)
        diff2 = X[tup[0]]-X[tup[2]]
        diff2_tree = X0_tree - X2_tree
        diff2_all = np.dot(diff2,L.T)+diff2_tree
        if 1 + np.sum(diff_all**2) - np.sum(diff2_all**2) > 0:
            gradient += reg*(np.outer(diff_all, diff_all) - np.outer(diff2_all, diff2_all))
        return gradient
