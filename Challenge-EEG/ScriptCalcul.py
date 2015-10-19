
import numpy as np
import pandas as pd
from loadAndCreateFeatures import (loadData,featureCreation,windowing2)
from AnalyzeEEG import createConfusionMatrix,createRocCurve
from sklearn.svm import SVC
from extractFeaturesSource import normalize,dataTransform,normalizeTest

############### Window of size 3500 , crossing of 1000

X_train,y_train,X_test = loadData()

dico = {'W ':0,'N1':1,'N2':2,'N3':3,'R ':4}
y_train_np=np.array([dico[y] for y in y_train])


print 'extracting train features'
X_train_Df = featureCreation(X_train,4,30)

print 'extracting window 2 train features'
XW2,XW2_Df,y2t = windowing2(X_train,y_train_np)

print 'extracting test features'
X_test_Df = featureCreation(X_test,4,30)

print 'extracting window 2 test features'
XW2t,XW2t_Df,_ = windowing2(X_test,[])

X_train_DftN = normalize(dataTransform(X_train_Df))
X_test_DftN = normalizeTest(dataTransform(X_test_Df),dataTransform(X_train_Df))
XW2_DftN = normalize(dataTransform(XW2_DftN))
XW2t_DftN = normalizeTest(dataTransform(XW2t_Df),dataTransform(XW2_Df))

colx = (['P_rel_beta', 'WAMP', 'S6', 'STD_rel_beta', 'AMP_rel_sigma', 'Hfd', 'Kurtosis',
        'Hurst', 'Spectral_Entropy', 'AP_rel_sigma', 'P_rel_epsilon', 'dasd2', 'S3',
        'AP_rel_epsilon', 'E10', 'Dfa', 'STD', 'AP_rel_beta', 'E5', 'Pfd', 'ap_entropy',
        'P_rel_alpha', 'E9', 'M3', 'WSstd', 'M1', 'complexity', 'E6', 'S5', 'cwtb4',
        'kol', 'WSstdB6rbio', 'STD_rel_alpha', 'WSTB1rbio', 'WSstdB2db20', 'WSTB5sym'])

col2x = (['P_rel_beta', 'ap_entropy', 'peaks', 'AMP_rel_sigma', 'P_rel_epsilon', 'Hurst',
          'Spectral_Entropy', 'WSstdB3db20', 'Kurtosis', 'WAMP', 'AP_rel_sigma', 'WSstdB5rbio',
          'AP_rel_beta', 'STD', 'dasd2', 'E5', 'WSTB6sym', 'STD_rel_epsilon', 'cwtb4',
          'Hfd', 'WSTB2rbio'])

SVCt = SVC(gamma=0.0071000,C = 5.674999)
SVCt.fit(X_train_DftN[colx],y_train_np)
predict=np.array(SVCt.predict(X_test_DftN[colx]))
SVCt2 = SVC(C= 9.0, gamma= 0.00800)
SVCt2.fit(XW2_DftN[col2x],y2t)
predict2=np.array(SVCt2.predict(XW2t_DftN[col2x]))
n=len(predict)
t1 = np.array([i%2==0 for i in range(2*n)])
t2 = np.array([i%2==1 for i in range(2*n)])

result_Df = pd.DataFrame(predict,columns=['all'])
result_Df['t1']=predict2[t1]
result_Df['t2']=predict2[t2]

def majorityVote(line):
    res=np.zeros(5)
    res[line['all']] = 1
    res[line['t1']] = 0.5
    res[line['t2']] = 0.5
    return np.argmax(res)

result_Df['vote2'] = result_Df.apply(lambda x : majorityVote(x),axis=1)
