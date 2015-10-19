import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import io
import os
import sys
import time
import numpy
import numpy as np
import theano
import bisect
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
from mlp import MLP

from math import sqrt

from scipy.sparse.linalg import svds
from scipy.linalg import svd

from sklearn.metrics.pairwise import rbf_kernel


def rank_trunc(gram_mat, k, fast=True):
    """
    k-th order approximation of the Gram Matrix G.

    Parameters
    ----------
    gram_mat : array, shape (n_samples, n_samples)
        the Gram matrix
    k : int
        the order approximation
    fast : bool
        use svd (if False) or svds (if True).

    Return
    ------
    gram_mat_k : array, shape (n_samples, n_samples)
        The rank k Gram matrix.
    """
    if fast:
        u,s,v = svds(gram_mat,k=k)
        gram_mat_k = np.dot(u,np.dot(np.diag(s),v))
    else:
        u,s,v = svd(gram_mat)
        gram_mat_k = np.dot(u[:,:k],np.dot(np.diag(s[:k]),v[:k,:]))
    return gram_mat_k


def random_features(X_train, X_test, gamma, c=300, seed=44):
    """Compute random kernel features

    Parameters
    ----------
    X_train : array, shape (n_samples1, n_features)
        The train samples.
    X_test : array, shape (n_samples2, n_features)
        The test samples.
    gamma : float
        The Gaussian kernel parameter
    c : int
        The number of components
    seed : int
        The seed for random number generation

    Return
    ------
    X_new_train : array, shape (n_samples1, c)
        The new train samples.
    X_new_test : array, shape (n_samples2, c)
        The new test samples.
    """
    rng = np.random.RandomState(seed)
    W = np.sqrt(2.0*gamma)*rng.randn(X_train.shape[1],c)
    b = rng.uniform(0,2*np.pi,size=c)
    X_new_train = np.sqrt(2.0/c)*np.cos(np.dot(X_train,W) + b)
    X_new_test = np.sqrt(2.0/c)*np.cos(np.dot(X_test,W) + b)
    return X_new_train, X_new_test

class nystrom():
    def __init__ (self,gamma, c=500, k=200, seed=44):
        """ Parameters 
            ----------
            gamma : float
            The Gaussian kernel parameter
            c : int
            The number of points to sample for the approximation
            k : int
            The number of components
            seed : int
            The seed for random number generation"""
        
        self.gamma = gamma
        self.c = c 
        self.k = k
        self.seed =seed 
        
        
    def fit_transform(self, X_train):
        """Compute nystrom kernel approximation

        Parameters
        ----------
        X_train : array, shape (n_samples1, n_features)
            The train samples.
        X_test : array, shape (n_samples2, n_features)
            The test samples.

        Return
        ------
        X_new_train : array, shape (n_samples1, c)
            The new train samples.
        """
        gamma = self.gamma
        c = self.c
        k = self.k
        seed = self.seed
        rng = np.random.RandomState(seed)
        I = rng.randint(0,X_train.shape[0],size=c)
        G = rbf_kernel(X_train[I],X_train[I],gamma)
        u,s,v = svd(G)
        del G
        #Gk = np.dot(u,np.dot(np.diag(s),v))
        s = s[:k]
        u = u[:,:k]
        #Gk = np.dot(u,np.dot(np.diag(s),v))
        s[s!=0.0] = np.power(s[s!=0.0],-0.5)
        self.Mk = np.dot(u,np.diag(s))
        self.X_trainI = X_train[I]

        T_train = rbf_kernel(X_train,X_train[I])
        #T_test = rbf_kernel(X_test,X_train[I])

        X_new_train = np.dot(T_train,self.Mk)
        #X_new_test = np.dot(T_test,Mk)            
        return X_new_train
    
    
    def fit(self, X_train):
        """Compute nystrom kernel approximation

        Parameters
        ----------
        X_train : array, shape (n_samples1, n_features)
            The train samples.
        Return
        ------
        None
        """
        gamma = self.gamma
        c = self.c
        k = self.k
        seed = self.seed
        rng = np.random.RandomState(seed)
        I = rng.randint(0,X_train.shape[0],size=c)
        G = rbf_kernel(X_train[I],X_train[I],gamma)

        u,s,v = svd(G)
        #Gk = np.dot(u,np.dot(np.diag(s),v))
        s = s[:k]
        u = u[:,:k]
        s[s!=0.0] = np.power(s[s!=0.0],-0.5)
        del G
        self.Mk = np.dot(u,np.diag(s))
        self.X_trainI = X_train[I]



        
    def transform(self, X_test):
        """
        Parameters
        ----------
        X_test : array, shape (n_samples1, n_features)
            The test samples.
            
        Return
        ------
        X_new_test : array, shape (n_samples2, c)
            The new test samples.
        
        """
        
        T_test = rbf_kernel(X_test,self.X_trainI)
        X_new_test = np.dot(T_test,self.Mk)  
        
        return X_new_test
        
def create_tuples(n,labels2,allLabels,X):
    m = labels2.shape[0]
    firstLabels = np.random.randint(0,m,n)
    indexes = np.arange(X.shape[0])
    data = []
    
    for i in range(0,n):
        
        idx_labels = indexes[allLabels == labels2[firstLabels[i]]]
        idx =  np.random.randint(0,len(idx_labels),2)
        idx_nolabels = indexes[allLabels != labels2[firstLabels[i]]]
        idx2 = np.random.randint(0,len(idx_nolabels),1)
        X0 = X[idx_labels[idx[0]]]
        X1 = X[idx_labels[idx[1]]]
        X2 = X[idx_nolabels[idx2][0]]
        data.append(np.r_[ (X0 + X1)/2 , np.abs(X0 - X1)])
        #r = np.sum(np.abs(X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]]))
        data.append(np.r_[ (X0 + X2)/2,np.abs(X0 - X2)])
        
    data = np.array(data)
    labels = [1,0]*n
    labels = np.array(labels)
    return data, labels
 
    
def generate_pairs_data(label,X, n_pairs, positive_ratio, random_state=42):
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
    data =[]
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
    for i in range(n_pairs):
        data.append(np.r_[X[pairs_idx[i,0]]+X[pairs_idx[i,1]],np.abs(X[pairs_idx[i,0]]-X[pairs_idx[i,1]])])
    data = np.array(data)
    
    return data,pairs_idx, pairs_label




    
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
rng = numpy.random.RandomState(1234)
learning_rate=0.01 
L1_reg=0.00
L2_reg=0.0001 
n_epochs=1000
n_hidden = 50

classifier = MLP(
        rng=rng,
        input=x,
        n_in=2000,
        n_hidden=n_hidden,
        n_out=2
    )

cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

validate_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y)
    )

test_model = theano.function(
        inputs=[x],
        outputs=classifier.y_pred
    )

test_model_proba = theano.function(
        inputs=[x],
        outputs=classifier.p_y_given_x
    )

# start-snippet-5
# compute the gradient of cost with respect to theta (sotred in params)
# the resulting gradients will be stored in a list gparams
gparams = [T.grad(cost, param) for param in classifier.params]

# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs

# given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
# same length, zip generates a list C of same size, where each element
# is a pair formed from the two lists :
#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]
# compiling a Theano function `train_model` that returns the cost, but
# in the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(
    inputs=[x,y],
    outputs=cost,
    updates=updates
)
# end-snippet-5

###############
# TRAIN MODEL #
###############

print 'loading Data'
train_facile = io.loadmat('train.mat')
X = train_facile['X']
Y = train_facile['label'].flatten()
        
print 'kernelization part'       
nys = nystrom(1.0/1500,5000,1000)
X = nys.fit_transform(X)

print 'create validation set'
#labels2 = np.loadtxt('labelfacile.txt')
#xval , yval = create_tuples(2000,labels2,Y,X)
xval , _ ,yval = generate_pairs_data(Y,X, 4000, 0.4, random_state=42)


print '... training'

start_time = time.clock()

epoch = 0

t1 = time.time()
while (epoch < 1000000):
    epoch = epoch + 1
    #xtrain, ytrain = create_tuples(10,labels2,Y,X)
    xtrain,_,ytrain = generate_pairs_data(Y,X, 20, 0.3, random_state=42)
    minibatch_avg_cost = train_model(xtrain.astype(np.float32),ytrain.astype(np.int32))
    if epoch % 1000 == 0:
        validation_loss = validate_model(xval.astype(np.float32),yval.astype(np.int32))
        print "validation loss: %0.5f" %validation_loss, "in %d iters" %epoch
    if epoch % 10000 == 0:
        yprd2 = test_model_proba(xval)
        fpr, tpr, thresholds = metrics.roc_curve(yval.astype(np.int32), yprd2[:,1])
        score_facile = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
        print "validation loss for metrics: %0.5f" %score_facile
        #idx = (np.abs(fpr + tpr - 1.)).argmin()
        #score_difficile = (fpr[idx]+(1-tpr[idx]))/2
        #print "validation loss for metrics: %0.5f" %score_difficile
        print "time elapse for 10000 iteration %0.3fs" %(time.time() - t1)
        t1 = time.time()

print "trainning ended" 

test_facile = io.loadmat('test.mat')
print(test_facile.keys())
Xt = test_facile['X']
Xtest = []
for pairs in test_facile['pairs']:
    X0 = nys.transform(Xt[pairs[0]])[0]
    X1 = nys.transform(Xt[pairs[1]])[0]
    xtest = np.r_[(X0 + X1)/2 ,np.abs(X0 - X1)]
    Xtest.append(xtest)
#
Xtest = np.array(Xtest)
yprd = test_model_proba(Xtest.astype(np.float32))
#
np.savetxt('soumission_neuronal_aws.txt', yprd[:,0], fmt='%.5f')



