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
    data = []
    
    for i in range(0,n):
        idx_labels = indexes[allLabels == labels2[firstLabels[i]]]
        idx =  np.random.randint(0,len(idx_labels),2)
        idx_nolabels = indexes[allLabels != labels2[firstLabels[i]]]
        idx2 = np.random.randint(0,len(idx_nolabels),1)
        
        #r = np.sum(np.abs(X[idx_labels[idx[0]]] + X[idx_labels[idx[1]]]))
        #data = np.r_[data , [np.r_[ (X[idx_labels[idx[0]]] + X[idx_labels[idx[1]]])/r, np.abs(X[idx_labels[idx[0]]] - X[idx_labels[idx[1]]])/r, np.multiply(X[idx_labels[idx[0]]],X[idx_labels[idx[1]]])]]]
        data.append(np.r_[ X[idx_labels[idx[0]]] + X[idx_labels[idx[1]]] , np.abs(X[idx_labels[idx[0]]] - X[idx_labels[idx[1]]])])
        #r = np.sum(np.abs(X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]]))
        #data = np.r_[data, [np.r_[ (X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]])/r, np.abs(X[idx_labels[idx[0]]] - X[idx_nolabels[idx2][0]])/r, np.multiply(X[idx_labels[idx[0]]],X[idx_nolabels[idx2][0]])]]]
        data.append(np.r_[ X[idx_labels[idx[0]]] + X[idx_nolabels[idx2][0]] ,np.abs(X[idx_labels[idx[0]]] - X[idx_nolabels[idx2][0]])])
        
    data = np.array(data)
    labels = [1,0]*n
    labels = np.array(labels)
    return data, labels

def create_pos_neg(n,labels3,labels2,allLabels,X):
    m = labels3.shape[0]
    data = []
    Labels = np.random.randint(0,m,2*n)
    indexes = np.arange(X.shape[0])
    for i in range(0,n):
        idx = indexes[allLabels == labels3[Labels[2*i]]][0]
        idx2 = indexes[allLabels == labels3[Labels[2*i+1]]][0]
        data.append(np.r_[ X[idx] + X[idx2] , np.abs(X[idx] - X[idx2])])
    data = np.array(data)
    labels = [0]*n
    labels = np.array(labels)
    return data, labels
        
    
    

print 'loading data'
train_facile = io.loadmat('train.mat')
X = train_facile['X']
Xm = X.mean(axis=0)
Xs = X.std(axis=0)
X = (X -Xm) / Xs
Y = train_facile['label'].flatten()

#train_difficile = io.loadmat('traind.mat')
#X2 = train_difficile['X']
#Y2 = np.array(train_difficile['label']).flatten()

#X = np.r_[X,X2]
#Y = np.r_[Y,Y2]

del train_facile
#del train_difficile
#del X2

    
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
l = T.scalar('l')
rng = numpy.random.RandomState(1234)
L1_reg=0.00
L2_reg=0.0001 
n_epochs=1000
n_hidden = 50

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
    (param, param - l * gparam)
    for param, gparam in zip(classifier.params, gparams)
]
# compiling a Theano function `train_model` that returns the cost, but
# in the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(
    inputs=[x,y,l],
    outputs=cost,
    updates=updates
)
# end-snippet-5

###############
# TRAIN MODEL #
###############
print '... training'


start_time = time.clock()

epoch = 0
done_looping = False
#labels = Y
#bb = np.array([ len(labels[labels == l])>1 for l in labels])
#labels2 = np.unique(labels[bb])
labels2 = np.loadtxt('labelfacile.txt')
labels3 = np.loadtxt('labelfacile2.txt')
#labels2 = np.r_[labels2,np.unique(Y2)]
#del Y2
xval , yval = create_tuples(1500,labels2,Y,X)
#xval2, yval2 = create_neg(1500,labels3,Y,X)
#xval = np.r_[xval1,xval2]
#yval = np.r_[yval1,yval2]
t1 = time.time()
learn = 0.02
patience = 3
k = 0
best = np.inf
while (epoch < 1000000) and (not done_looping):
    epoch = epoch + 1
    xtrain, ytrain = create_tuples(10,labels2,Y,X)
    #xtrain2, ytrain2 = create_neg(10,labels3,Y,X)
    #xtrain = np.r_[xtrain1,xtrain2]
    #ytrain = np.r_[ytrain1,ytrain2]
    minibatch_avg_cost = train_model(xtrain.astype(np.float32),ytrain.astype(np.int32),learn)
    if epoch % 1000 == 0:
        validation_loss = validate_model(xval.astype(np.float32),yval.astype(np.int32))
        print "validation loss: %0.5f" %validation_loss, "in %d iters" %epoch
    if epoch % 5000 == 0:
        yprd2 = test_model_proba(xval)
        fpr, tpr, thresholds = metrics.roc_curve(yval.astype(np.int32), yprd2[:,1])
        score_facile = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
        print "validation loss for metrics: %0.5f" %score_facile, " in %0.3fs" %(time.time() - t1), "current patience : %d" %patience
        if score_facile >= best:
            patience -=1
        else:
            best = score_facile
            patience = 3 + k*2
        if patience == 0:
            learn /= 2
            print "divide learn, learn rate : %0.6f" %learn
            k += 1
            patience = 3 + k*2
        if learn < 1e-4:
            done_looping = True
        #idx = (np.abs(fpr + tpr - 1.)).argmin()
        #score_difficile = (fpr[idx]+(1-tpr[idx]))/2
        #print "validation loss for metrics: %0.5f" %score_difficile
        t1 = time.time()

print "trainning ended" 

test_facile = io.loadmat('test.mat')
print(test_facile.keys())
Xt = test_facile['X']
Xt = (Xt-Xm)/Xs
Xtest = []
for pairs in test_facile['pairs']:
    xtest =  np.r_[Xt[pairs[0]] + Xt[pairs[1]] ,np.abs(Xt[pairs[0]] - Xt[pairs[1]])]
    #xtest = np.r_[Xt[pairs[0]] + Xt[pairs[1]] ,np.abs(Xt[pairs[0]] - Xt[pairs[1]]),np.multiply(Xt[pairs[0]],Xt[pairs[1]])]
    Xtest.append(xtest)
#
Xtest = np.array(Xtest)
yprd = test_model_proba(Xtest.astype(np.float32))
#
np.savetxt('soumission_neuronal_aws2.txt', yprd[:,0], fmt='%.5f')


#test_difficile = io.loadmat('testd.mat')

#Xtd = test_difficile['X']

#Xtestd = []

#for pairs in test_difficile['pairs']:
#    r = np.sum(np.abs(Xtd[pairs[0]] + Xtd[pairs[1]]))
#    #xtestd = np.r_[(Xtd[pairs[0]] + Xtd[pairs[1]])/r ,np.abs(Xtd[pairs[0]] - Xtd[pairs[1]])/r]
#    xtestd = np.abs(Xtd[pairs[0]] - Xtd[pairs[1]])/r
#    Xtestd.append(xtestd)

#yprdd = test_model_proba(Xtestd)

#np.savetxt('soumission_neuronal_aws.txt', yprdd[:,0], fmt='%.5f')
