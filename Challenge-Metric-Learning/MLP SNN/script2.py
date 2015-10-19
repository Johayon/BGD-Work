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

def create_tuples(n,labels2,allLabels,X):
    m = labels2.shape[0]
    firstLabels = np.random.randint(0,m,n)
    indexes = np.arange(X.shape[0])
    data = np.array([])
    
    idx_labels = indexes[allLabels == labels2[firstLabels[0]]]
    idx =  np.random.randint(0,len(idx_labels),2)
    idx_nolabels = indexes[allLabels != labels2[firstLabels[0]]]
    idx2 = np.random.randint(0,len(idx_nolabels),1)
    data = [np.r_[ X[idx_labels[idx[0]]] + X[idx_labels[idx[1]]], np.abs(X[idx_labels[idx[0]]] - X[idx_labels[idx[1]]])]]
    data = np.r_[data, [np.r_[ X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]], np.abs(X[idx_labels[idx[0]]] - X[idx_nolabels[idx2][0]])]]]
    
    for i in range(1,n):
        idx_labels = indexes[allLabels == labels2[firstLabels[i]]]
        idx =  np.random.randint(0,len(idx_labels),2)
        idx_nolabels = indexes[allLabels != labels2[firstLabels[i]]]
        idx2 = np.random.randint(0,len(idx_nolabels),1)
        data = np.r_[data, [np.r_[ X[idx_labels[idx[0]]] + X[idx_labels[idx[1]]], np.abs(X[idx_labels[idx[0]]] - X[idx_labels[idx[1]]])]]]
        data = np.r_[data, [np.r_[ X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]], np.abs(X[idx_labels[idx[0]]] - X[idx_nolabels[idx2][0]])]]]                                
    
    labels = [1,0]*n
    labels = np.array(labels)
    return data, labels



train_facile = io.loadmat('train.mat')
X = train_facile['X']
Y = np.array(train_facile['label']).flatten()
    
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
rng = numpy.random.RandomState(1234)
learning_rate=0.01 
L1_reg=0.00 
L2_reg=0.0001 
n_epochs=1000
dataset='train.mat' 
batch_size=1000 
n_hidden=50

classifier = MLP(
        rng=rng,
        input=x,
        n_in=3000,
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
print '... training'

# early-stopping parameters
patience = 20  # look as this many examples regardless
patience_increase = 4  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
    
best_validation_loss = numpy.inf
best_iter = 0
start_time = time.clock()

epoch = 0
done_looping = False

labels = Y
bb = np.array([ len(labels[labels == l])>1 for l in labels])
labels2 = np.unique(labels[bb])

xval , yval = create_tuples(2000,labels2,Y,X)
t1 = time.time()
while (epoch < 1000000) and (not done_looping):
    epoch = epoch + 1
    xtrain, ytrain = create_tuples(10,labels2,Y,X)
    minibatch_avg_cost = train_model(xtrain,ytrain.astype(np.int32))
    if epoch % 1000 == 0:
        validation_loss = validate_model(xval,yval.astype(np.int32))
        print "validation loss: %0.5f" %validation_loss, "in %d iters" %epoch
    if epoch % 10000 == 0:
        yprd2 = test_model_proba(xval)
        fpr, tpr, thresholds = metrics.roc_curve(yval.astype(np.int32), yprd2[:,1])
        score_facile = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
        print "validation loss for metrics: %0.5f" %score_facile
        print "time elapse for 10000 iteration %0.3fs" %(time.time() - t1)
        t1 = time.time()

        
test_facile = io.loadmat('test.mat')
print(test_facile.keys())

Xt = test_facile['X']

Xtest = []
for pairs in test_facile['pairs']:
    xtest = np.r_[Xt[pairs[0]] + Xt[pairs[1]] ,np.abs(Xt[pairs[0]] - Xt[pairs[1]])]
    Xtest.append(xtest)
    
yprd = test_model_proba(Xtest)

np.savetxt('soumission_neuronal_aws.txt', yprd[:,0], fmt='%.5f')
