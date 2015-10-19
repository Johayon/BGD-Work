# -*- coding: utf-8 -*-

"""

Created on Thu 12 mar 2015
    
@author: Ohayon

"""

import numpy as np
from scipy import io
import pandas as pd
import sklearn as sk
import random
from extractFeaturesSource import (normalize,dataTransform,addFractalInfo,normalizeTest,
									addFftInfo,addWaveletInfo,addGlobalInfo,addWaveletInfoV2,addFilteredInfo,addWaveletInfoV3)
from preprocess import preprocessSignal,waveletpreprocess,waveletpre

###########################################################################################
#																						  #
#																						  #
""" 							Load data from mat to dataframe							"""
#																						  #
#																						  #
###########################################################################################

def loadData():
	dataset = io.loadmat('data_challenge.mat')
	dataset['X_train'].shape
	#dataset.keys()
	X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']
	return X_train, y_train, X_test

###########################################################################################
#																						  #
#																						  #
""" 							Randomize data											"""
#																						  #
#																						  #
###########################################################################################

def shuffleData(X,y):
	df = X.copy()
	df['Label'] = y
	shuffleDf = df.reindex(np.random.permutation(df.index))
	shuffleDf.reset_index(inplace=True)
	shuffleDf.drop('index',axis=1,inplace=True)
	return shuffleDf.drop('Label',axis=1),shuffleDf['Label']

def subsampling(x,n):
	return x.ix[random.sample(x.index, n)]

###########################################################################################
#																						  #
#																						  #
""" 							Extract Features										"""
#																						  #
#																						  #
###########################################################################################

def addFFTFeatures(X_data):
	features=(['Power','P_rel_delta','P_rel_theta','P_rel_alpha','P_rel_sigma','P_rel_beta', 'P_rel_epsilon',
				'AMP','AMP_rel_delta','AMP_rel_theta','AMP_rel_alpha','AMP_rel_sigma','AMP_rel_beta', 'AMP_rel_epsilon',
				'STD','STD_rel_delta','STD_rel_theta','STD_rel_alpha','STD_rel_sigma','STD_rel_beta', 'STD_rel_epsilon',
				'APower','AP_rel_delta','AP_rel_theta','AP_rel_alpha','AP_rel_sigma','AP_rel_beta','AP_rel_epsilon'])
	X_data_temp=[addFftInfo(X) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
	return X_data_Df

def addWaveFeatures(X_data):
	features=(['WRMS','WRMS_rel_delta','WRMS_rel_theta','WRMS_rel_alpha','WRMS_rel_sigma','WRMS_rel_beta',
				'WaveLengthA4','WaveLengthD4','WaveLengthD3','WaveLengthD2','WaveLengthD1','TotalRMS'])
	X_data_temp=[addWaveletInfo(X) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
	return X_data_Df

def addWaveFeaturesV2(X_data,level,window):
	features=([ 'E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','M1','M2','M3','M4','M5','M6','S1','S2','S3',
		'S4','S5','S6','MYOP','WAMP','ZC','MFL','MAV','VAR','WSstd','WSTdb2','WSTrbio','WSTsym','WSstdB1db20','WSstdB2db20','WSstdB3db20',
		'WSstdB4db20','WSstdB5db20','WSstdB6db20','WSstdB1rbio','WSstdB2rbio','WSstdB3rbio',
		'WSstdB4rbio','WSstdB5rbio','WSstdB6rbio','WSTB1rbio','WSTB2rbio','WSTB3rbio','WSTB4rbio','WSTB5rbio','WSTB6rbio', 
		'WSTB1sym','WSTB2sym','WSTB3sym','WSTB4sym','WSTB5sym','WSTB6sym','cwtb1','cwtb2','cwtb3','cwtb4','cwtb5'])
	X_data_temp=[addWaveletInfoV2(X,level,window) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
	return X_data_Df

def addWaveFeaturesV3(X_data,level,window):
 	features=([ 'E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','M1','M2','M3','M4','M5','M6','S1','S2','S3',
 		'S4','S5','S6','MYOP','WAMP','ZC','MFL','MAV','VAR','WSstd','WSTdb2','WSTrbio','WSTsym','WSstdB1db20','WSstdB2db20','WSstdB3db20',
 		'WSstdB4db20','WSstdB5db20','WSstdB6db20','WSstdB1rbio','WSstdB2rbio','WSstdB3rbio',
 		'WSstdB4rbio','WSstdB5rbio','WSstdB6rbio','WSTB1rbio','WSTB2rbio','WSTB3rbio','WSTB4rbio','WSTB5rbio','WSTB6rbio', 
 		'WSTB1sym','WSTB2sym','WSTB3sym','WSTB4sym','WSTB5sym','WSTB6sym','cwtb1','cwtb2','cwtb3','cwtb4','cwtb5'])
 	X_data_temp=[addWaveletInfoV3(X,level,window) for X in X_data]
 	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
 	return X_data_Df


def addGlobalFeatures(X_data):
	features = ['Fisher_Info','SVD_Entropy','Spectral_Entropy','Kurtosis','Skewness','AAC','DASD','peaks','zeros']
	X_data_temp=[addGlobalInfo(X) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
	return X_data_Df

def addFractalFeatures(X_data):
	fractabol = ['Hurst','Pfd','Hfd','Dfa','mobility','complexity','morbidity']
	X_data_fractal=[addFractalInfo(X) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_fractal,columns=fractabol)
	return X_data_Df

def addFilteredFeatures(X_data):
	features = ['ap_entropy','aac2','dasd2','std2','kol']
	X_data_temp = [addFilteredInfo(X) for X in X_data]
	X_data_Df=pd.DataFrame(X_data_temp,columns=features)
	return X_data_Df

def labelToNumber(y_train):
	dictionnary = {'W ':0,'N1':1,'N2':2,'N3':3,'R ':4}
	#revdictionnary = {'0': 'W', '1':'N1','2':'N2','3':'N3','4':'R'}
	y_train_np = np.array([dictionnary[y] for y in y_train])
	return y_train_np



##### Main Function



def featureCreation(X_list,level,window):
	print 'extracting Features'

	print 'extracting FFT features'
	X_Df = addFFTFeatures(X_list)

	print 'extracting Wavelet features'
	#X_train_temp = addWaveFeatures(X_train_pre)
	X_train_temp = addWaveFeaturesV2(X_list,level,window)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting Global features'
	X_train_temp = addGlobalFeatures(X_list)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting Fractal features (slow)'
	X_train_temp = addFractalFeatures(X_list)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting filtered features'
	X_list_pre = [waveletpre(X,level) for X in X_list]
	X_train_temp = addFilteredFeatures(X_list_pre)
	X_Df = X_Df.join(X_train_temp)

	return X_Df

def featureCreation2(X_list,level,window):
	print 'extracting Features'

	print 'extracting FFT features'
	X_Df = addFFTFeatures(X_list)

	print 'extracting Wavelet features'
	#X_train_temp = addWaveFeatures(X_train_pre)
	X_train_temp = addWaveFeaturesV3(X_list,level,window)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting Global features'
	X_train_temp = addGlobalFeatures(X_list)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting Fractal features (slow)'
	X_train_temp = addFractalFeatures(X_list)
	X_Df = X_Df.join(X_train_temp)

	print 'extracting filtered features'
	X_list_pre = [waveletpre(X,level) for X in X_list]
	X_train_temp = addFilteredFeatures(X_list_pre)
	X_Df = X_Df.join(X_train_temp)

	return X_Df



def windowing2(X_list,y):
	print 'Window creation'
	X_s1 = np.array([X[:3500] for X in X_list])
	X_s2 = np.array([X[2500:] for X in X_list])

	X_W2=np.zeros((2*len(X_list),3500))
	for i in range(len(X_list)):
		X_W2[2*i,:]=X_s1[i,:]
		X_W2[2*i+1,:]=X_s2[i,:]

	print 'Window features extraction'
	X2D = featureCreation2(X_W2,4,15)
	y2t=[]

	for i in range(len(y)):
		y2t.append(y[i])
		y2t.append(y[i])

	return X_W2,X2D,y2t



def windowing4(X_list,y):
	print 'Window creation'
	X_f1 = np.array([X[:1875] for X in X_list])
	X_f2 = np.array([X[1375:3250] for X in X_list])
	X_f3 = np.array([X[2750:4625] for X in X_list])
	X_f4 = np.array([X[4125:] for X in X_list])

	X_W4=np.zeros((4*len(X_list),1875))
	
	for i in range(len(X_list)):
		X_W4[4*i,:]=X_f1[i,:]
		X_W4[4*i+1,:]=X_f2[i,:]
		X_W4[4*i+2,:]=X_f3[i,:]
		X_W4[4*i+3,:]=X_f4[i,:]

	print 'Window features extraction'
	X4D = featureCreation(X_W4,4)

	Y4t=[]

	for i in range(len(y)):
		Y4t.append(y[i])
		Y4t.append(y[i])
		Y4t.append(y[i])
		Y4t.append(y[i])

	return X_W4,X4D,Y4t

def createOutput(res,output):
	revdictionnary = {'0': 'W', '1':'N1','2':'N2','3':'N3','4':'R'}
	result = [revdictionnary[str(k)] for k in res]

	f = open('Result/'+ output,'w')
	for r in result:
		f.write(r+'\n')
	f.close()
	return res

































