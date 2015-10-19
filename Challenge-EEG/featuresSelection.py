from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def greedy(clf,X,y,cv=3,start=[],start2=[],be=0):
	features_selected = start
	if len(start2) > 0:
		features_to_test = start2
	else:
		features_to_test = X.columns.values
	selection = True
	best_score = be
	while selection:
		features_score=[]
		for feature in features_to_test:
			cross_features = np.concatenate([features_selected,[feature]])
			print cross_features
			score =cross_val_score(clf,X[cross_features],y,cv=cv)
			features_score.append(np.mean(score))
		mscore = np.max(features_score)
		find = np.argmax(features_score)
		if mscore >= best_score:
			best_score = mscore
			features_selected.append(features_to_test[find])
			features_to_test=np.delete(features_to_test,find)
			print best_score,features_selected
		else:
			selection=False
	return best_score,features_selected

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def featureSelection(X,y):
	class RandomForestClassifierWithCoef(RandomForestClassifier):
	    def fit(self, *args, **kwargs):
	        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
	        self.coef_ = self.feature_importances_
	randfor = RandomForestClassifierWithCoef(n_estimators=35)
	rfecv = RFECV(estimator=randfor, step=1, cv=5,
	               scoring='accuracy',verbose=2)
	rfecv.fit(X,y)
	return X.columns[rfecv.get_support()]

from sklearn.ensemble import ExtraTreesClassifier

def plotImportance(X,y):
	forest = ExtraTreesClassifier(n_estimators=250,
	                              random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]
	n=X.shape[1]

	#Print the feature ranking
	#print("Feature ranking:")

	#for f in range(n):
	#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure(figsize=(20,15))
	plt.title("Feature importances")
	plt.bar(range(n), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(n), X.columns[indices],rotation=90)
	plt.xlim([-1, n])
	plt.savefig('featuresel.pdf')

	#return indices,importances



def createConfusionMatrix(predict,y_test1):
	y_test=np.array(y_test1)
	cm = confusion_matrix(y_test, predict)
	plot_confusion_matrix(cm)

def plot_confusion_matrix(cm, title='Confusion matrix \n', cmap=plt.cm.Blues):
    sns.set_style("dark")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    width = len(cm)
    height = len(cm[0])
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cm_normalized), cmap=cmap, 
                interpolation='nearest')
    for x in xrange(width):
        for y in xrange(height):
            color = 'orange'
            if x == y:
                color = 'lightgreen'
            ax.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center',size=15,color=color)
    cb = fig.colorbar(res)
    
    
    plt.title(title)
    tick_marks = np.arange(len(['W','N1','N2','N3','R']))
    plt.xticks(tick_marks, ['W','N1','N2','N3','R'], rotation=45)
    plt.yticks(tick_marks, ['W','N1','N2','N3','R'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


