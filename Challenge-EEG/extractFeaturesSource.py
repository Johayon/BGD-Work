# -*- coding: utf-8 -*-

"""

Created on Thu 12 mar 2015
    
@author: Ohayon

"""

from numpy.fft import fft
import numpy as np
from numpy import floor
import pywt as pywt
from scipy import stats
from scipy.fftpack import rfft, irfft, fftfreq
import math
from Univariate import (hurst, pfd, hfd, dfa, hjorth, spectral_entropy, fisher_info, svd_entropy, ap_entropy, samp_entropy)
import mlpy.wavelet as wave


###########################################################################################
#																						  #
#																						  #
""" 							FFT feature related 									"""
#																						  #
#																						  #
###########################################################################################


"""
INPUTS = X      signal
		 Band   bande de frequence a analyser
		 Fs     Frequence d'echantillonage
OUTPUTS = Power           	puissance
		  Power_ratio     	ratio de puissance dans les bandes
		  Amp             	amplitude
		  amp_ratio       	ratio d'amplitude dans les bandes
		  STD             	Variance
		  STD_ratio 	  	ratio de la variance dans les bandes
		  absPowers       	puissance absolue
		  absPowers_Ratio	ratio de puissance absolue dans les bandes
"""

def extractFFT(X,Band,Fs):

	C = fft(X)
	C = np.abs(C)
	Power =np.zeros(len(Band)-1);
	absPowers = np.zeros(len(Band)-1);
	Amp = np.zeros(len(Band)-1);
	STD = np.zeros(len(Band)-1);
	for Freq_Index in xrange(0,len(Band)-1):
		Freq = float(Band[Freq_Index])										
		Next_Freq = float(Band[Freq_Index+1])
		Power[Freq_Index] = np.sum(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
		absPowers[Freq_Index] = (np.linalg.norm(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])**2)/len(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
		Amp[Freq_Index] = np.max(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
		STD[Freq_Index] = np.std(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
	Power_Ratio = Power/np.sum(Power)
	absPowers_Ratio = absPowers/np.sum(absPowers)
	Amp_Ratio = Amp/np.sum(Amp)
	STD_Ratio = STD/np.sum(STD)

	return Power,Power_Ratio,Amp,Amp_Ratio,STD,STD_Ratio,absPowers,absPowers_Ratio



def extractFFTV2(X,Band,Fs):
	C = fft(X)
	C = np.abs(C)
	Power =np.zeros(len(Band)-1)
	for Freq_Index in xrange(0,len(Band)-1):
		Freq = float(Band[Freq_Index])										
		Next_Freq = float(Band[Freq_Index+1])
		Power[Freq_Index] = np.sum(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
	Power_Ratio = Power/np.sum(Power)
	MPF = np.sum(Power * ((Band[:-1]+Band[1:])/2))/np.sum(Power)
	s = 0
	MDF=0.0
	for i in range(len(Band)-1):
		s+=Power_Ratio[i]
		if s >=0.5:
			MDF = Band[i+1]
			break
	MNP = np.sum(Power)/(len(Band)-1)
	k = np.argmax(Power)
	PKF = (Band[k] + Band[k+1]) /2
	SM1 = np.sum(Power * ((Band[:-1]+Band[1:])/2))
	SM2 = np.sum(Power * ((Band[:-1]+Band[1:])/2)**2)
	SM3 = np.sum(Power * ((Band[:-1]+Band[1:])/2)**3)
	FR = np.sum(Power[:5]) / np.sum(Power[-5:])
	VCF = SM2/np.sum(Power) - ((SM1)/np.sum(Power))**2
	return MPF,MDF,MNP,PKF,SM1,SM2,SM3,FR,VCF





def StftV2(X,wsize):
    Xst = np.abs(mne.time_frequency.stft(X,wsize))
    T = Xst.sum(axis=1)
    SEF1,SEF2 = np.zeros((10178,30)),np.zeros((10178,30))
    SEFd = np.zeros(10178)
    for k in range(10178):
        for i in range(30):
            cumsum = 0
            for j in range(Xst.shape[1]):
                if  cumsum/T[k,i] < 0.5:
                    SEF1[k,i]=j
                elif cumsum/T[k,i] < 0.95:
                    SEF2[k,i]=j
                cumsum += Xst[k,j,i]
    SEFd = np.mean(SEF2-SEF1,axis=1)
    return SEFd,SEF1,SEF2

def Stft(X,band):
    Xst = np.abs(mne.time_frequency.stft(X,200))
    res = np.zeros((len(X),len(band)-1,60))
    for j in range(len(X)):
        for i in range(len(band)-1):
            res[j,i,:] = np.sum(X[j,np.floor(band[i]):np.floor(band[i+1]),:],axis=0)
    RMS = np.linalg.norm(res,axis=2)/59
    Power = np.sum(res,axis=2)
    trend = np.apply_along_axis(lambda x : np.polyfit(x,np.linspace(0,30,60),1,full=False)[1],2,res)
    return RMS,Power,trend



###########################################################################################
#																						  #
#																						  #
"""  							Wavelet features related 								"""
#																						  #
#																						  #
###########################################################################################


def extractRMSVWavelet(signal):
	c_a,c_d4,c_d3,cd_2,cd_1 = pywt.wavedec(signal,'db8',mode='sym',level=4)
	RMS = np.array([math.sqrt(np.linalg.norm(wave)**2/(len(wave)-1)) for wave in [c_a,c_d4,c_d3,cd_2,cd_1]])
	Power = np.array([np.sum(np.abs(wave)) for wave in [c_a,c_d4,c_d3,cd_2,cd_1]])
	WaveLength = np.array([np.sum(np.abs(wave[:-1]-wave[1:])) for wave in [c_a,c_d4,c_d3,cd_2,cd_1]])
	relRMS = RMS/np.sum(RMS)
	relPower = Power/np.sum(Power)
	Tc=np.r_[c_a,c_d4,c_d3,cd_2,cd_1]
	TotalRMS = math.sqrt(np.linalg.norm(Tc)**2/(len(Tc)-1))
 	return RMS,relRMS,WaveLength,TotalRMS


def extractWaveletV2(signal,level):
	
	wp = pywt.WaveletPacket(data=signal, wavelet='db2', mode='sym')

	E=np.zeros(10)
	M=np.zeros(6)
	S=np.zeros(6)

	nodes = ([['aaaadd','aaaada','aaaaad','aaaaaad'],['aadada','aadaa','aaadd','aaadad','aaadaa'],
		['aadda','aadadd'],['adaad','adaaa','aaddd'],['adda','adad'],['da','addd']])
	for i in range(len(nodes)):
		C = np.concatenate([wp[node].data for node in nodes[i]])
		E[i] = (np.linalg.norm(C)**2)/len(C)
		M[i] = np.mean(np.abs(C))
		S[i] = np.std(C)
	E[6]=np.sum(E[0:6])
	E[7]=E[2]/(E[1]+E[0])
	E[8]=E[1]/(E[0]+E[2])
	E[9]=E[0]/(E[2]+E[1])

	res = np.concatenate([E,M,S])

	c = pywt.wavedec(signal,'db7',mode='sym',level=level)
	a=pywt.idwt(c[0],None,'db7')
	threshold=20#np.median(np.abs(a))
	MYOP = 1.0/len(a) * len(a[a>threshold])
	k=np.abs(a[:-1]-a[1:])
	l=a[:-1]*a[1:]
	WAMP = len(k[k>threshold])
	ZC= np.sum(k[l>0])
	MFL = np.log10(np.linalg.norm(k))
	MAV = np.sum(np.abs(a))/len(a)
	VAR = np.linalg.norm(k)**2/(len(a)-1)

	return res,MYOP,WAMP,ZC,MFL,MAV,VAR

def extractWaveletEntropy(signal,window,wave):
	wp = pywt.WaveletPacket(data=signal, wavelet=wave, mode='sym')

	E=np.zeros((6,window))
	TE = np.zeros(6)
	p=np.zeros((6,window))
	WS = np.zeros(window)
	nodes = ([['aaaadd','aaaada','aaaaad','aaaaaad'],['aadada','aadaa','aaadd','aaadad','aaadaa'],
		['aadda','aadadd'],['adaad','adaaa','aaddd'],['adda','adad'],['da','addd']])

	for i in range(len(nodes)):
		C = [wp[node].data for node in nodes[i]]
		for k in range(window):
			div=0
			for c in C:
				frame= len(c)/(window+1)
				E[i,k] += (np.linalg.norm(c[k*frame:(k+1)*frame])**2)/len(c[k*frame:(k+1)*frame])
	for i in range(6):
		TE[i] = np.sum(E[i,:])
		for j in range(window):
			p[i,j] = E[i,j]/TE[i]
	for j in range(window):
		WS[j] = - np.sum(p[:,j]*np.log2(p[:,j]))
	return np.std(WS)

def extractWaveletEntropy2(signal,window,wave):
	wp = pywt.WaveletPacket(data=signal, wavelet=wave, mode='sym')

	E=np.zeros((6,window))
	TE = np.zeros(6)
	p=np.zeros((6,window))
	WS = np.zeros((6,window))
	nodes = ([['aaaadd','aaaada','aaaaad','aaaaaad'],['aadada','aadaa','aaadd','aaadad','aaadaa'],
		['aadda','aadadd'],['adaad','adaaa','aaddd'],['adda','adad'],['da','addd']])

	for i in range(len(nodes)):
		C = [wp[node].data for node in nodes[i]]
		for k in range(window):
			div=0
			for c in C:
				frame= len(c)/(window+1)
				E[i,k] += (np.linalg.norm(c[k*frame:(k+1)*frame])**2)/len(c[k*frame:(k+1)*frame])
	for i in range(6):
		TE[i] = np.sum(E[i,:])
		p[i,:] = E[i,:]/TE[i]
	for j in range(window):
		WS[:,j] = - p[:,j]*np.log2(p[:,j])
	return np.std(WS,axis=1)

def extractWaveletTotalEntropy2(signal,wave):
	wp = pywt.WaveletPacket(data=signal, wavelet=wave, mode='sym')

	E=np.zeros(6)
	TE =0 
	p=np.zeros(6)
	WS = np.zeros(6)
	nodes = ([['aaaadd','aaaada','aaaaad','aaaaaad'],['aadada','aadaa','aaadd','aaadad','aaadaa'],
		['aadda','aadadd'],['adaad','adaaa','aaddd'],['adda','adad'],['da','addd']])

	for i in range(len(nodes)):
		C = [wp[node].data for node in nodes[i]]
		E[i] = np.mean([(np.linalg.norm(c)**2)/len(c) for c in C])
	TE = np.sum(E)
	p = E/TE
	WS = - p[:]*np.log2(p[:])
	return WS

def extractWaveletTotalEntropy(signal,wave):
	wp = pywt.WaveletPacket(data=signal, wavelet=wave, mode='sym')

	E=np.zeros(6)
	TE =0 
	p=np.zeros(6)
	WS = 0
	nodes = ([['aaaadd','aaaada','aaaaad','aaaaaad'],['aadada','aadaa','aaadd','aaadad','aaadaa'],
		['aadda','aadadd'],['adaad','adaaa','aaddd'],['adda','adad'],['da','addd']])

	for i in range(len(nodes)):
		C = [wp[node].data for node in nodes[i]]
		E[i] = np.mean([(np.linalg.norm(c)**2)/len(c) for c in C])
	TE = np.sum(E)
	p = E/TE
	WS = - np.sum(p[:]*np.log2(p[:]))
	return WS

def freqNotation(X_f):
	x = X_f
	scales = wave.autoscales(N=x.shape[0], dt=0.05, dj=0.07125, wf='morlet',p=0.5)
	X = np.real(wave.cwt(x=x, dt=0.05, scales=scales, wf='morlet',p=0.5))
	X1 = X[1:30]
	X2 = X[30:60]
	X3 = X[60:100]
	X4 = X[100:140]
	X5 = X[140:200]

	hist1 = np.sum(np.abs(X1),axis=0)
	hist2 = np.sum(np.abs(X2),axis=0)
	hist3 = np.sum(np.abs(X3),axis=0)
	hist4 = np.sum(np.abs(X4),axis=0)
	hist5 = np.sum(np.abs(X5),axis=0)

	H=np.concatenate([hist1,hist2,hist3,hist3,hist5])

	val = np.percentile(H,75)

	peaks1 = hist1.copy()
	peaks1[peaks1<val]=0
	peaks1[peaks1>=val]=1

	peaks2 = hist2.copy()
	peaks2[peaks2<val]=0
	peaks2[peaks2>=val]=1

	peaks3 = hist3.copy()
	peaks3[peaks3<val]=0
	peaks3[peaks3>=val]=1

	peaks4 = hist4.copy()
	peaks4[peaks4<val]=0
	peaks4[peaks4>=val]=1

	peaks5 = hist5.copy()
	peaks5[peaks5<val]=0
	peaks5[peaks5>=val]=1

	i1=np.sum(peaks1)/len(peaks1)
	i2=np.sum(peaks2)/len(peaks2)
	i3=np.sum(peaks3)/len(peaks3)
	i4=np.sum(peaks4)/len(peaks4)
	i5=np.sum(peaks5)/len(peaks5)

	return [i1,i2,i3,i4,i5]



###########################################################################################
#																						  #
#																						  #
""" 							Brut signal features related 							"""
#																						  #
#																						  #
###########################################################################################

def extractEntropy(signal,L):
	s0 = np.min(signal)
	sL = np.max(signal)
	k=np.array([math.floor((s-s0)/(sL-s0)*L) for s in signal])
	n=np.array([len(k[k==i]) for i in range(L)])
	n[L-1]=n[L-1]+1
	res = np.linalg.norm(n*1.0/6000)**2
	return 1-res

def extractMaxamplitude(signal):
	return np.max(signal)

def extractSTD(signal):
	return np.std(signal)

def averageAmplitudeChange(X):
	return np.sum(np.abs(X[:-1]-X[1:]))/len(X)

def differenceAbsoluteSTD(X):
	return np.sqrt((np.linalg.norm(X[:-1]-X[1:])**2)/(len(X)-1))

import zlib 
def kolmogorov(s):
  l = float(len(s))
  compr = zlib.compress(s)
  c = float(len(compr))
  return c/l 

def extractKolmogorov(X):
	val = np.percentile(np.abs(X),33)
	s = X.copy()
	s[np.abs(s)<val] = 0.0
	s[np.abs(s)>=val] = np.sign(s[np.abs(s)>=val])*1.0
	return kolmogorov(s)

def peaks(X):
    t = np.percentile(np.abs(X),50)
    m=(X[:-2]-X[1:-1]) * (X[1:-1]-X[2:])
    p = len(X[np.sign(m)<0][np.abs(X[np.sign(m)<0])>t])/30
    return p

def zeros(X):
    return len(X[np.abs(X) < 1])



###########################################################################################
#																						  #
#																						  #
""" 							Normalisation and Creation								"""
#																						  #
#																						  #
###########################################################################################


def normalize(X):
	return (X - np.mean(X,axis=0))/np.std(X,axis=0)

def normalizeTest(X_test,X_train):
	return (X_test - np.mean(X_train,axis=0))/np.std(X_train,axis=0)

def dataTransform(X):
	res = X.copy()
	for column in res.columns:
		if 'rel_delta' in column: #and not 'AMP' in column:
			res[column] = np.arcsin(np.sqrt(res[column]))
		elif 'rel_theta' in column: #and not 'AMP' in column:
			res[column] = np.arcsin(np.sqrt(res[column]))
		elif 'rel_alpha' in column: #and not 'AMP' in column:
			res[column] = np.log(res[column]/(1.0-res[column]))
		elif 'rel_sigma' in column: #and not 'AMP' in column:
			res[column] = np.log(res[column]/(1.0-res[column]))
		elif 'rel_beta' in column: #and not 'AMP' in column:
			res[column] = np.log(res[column]/(1.0-res[column]))
		elif 'rel_epsilon' in column: #and not 'AMP' in column:
			res[column] = np.log(res[column]/(1.0-res[column]))
	return res

def addFftInfo(X):
	Power,res1,AMP,res2,STD,res3,Apower,res5 = extractFFT(X,[0.5,4.5,8.5,12.5,16.5,32.5,48.5],200)
	return np.concatenate([[np.sum(Power)],res1,[np.sum(AMP)],res2,[np.sum(STD)],res3,[np.sum(Apower)],res5])

def addWaveletInfo(X):
	RMS,res1,res3,res4 = extractRMSVWavelet(X)
	return np.concatenate([[np.sum(RMS)],res1,res3,[res4]])

def addWaveletInfoV2(X,level,window):
	Energy,MYOP,WAMP,ZC,MFL,MAV,VAR = extractWaveletV2(X,level)
	Went = extractWaveletEntropy(X,window,'db2')
	WTentdb = extractWaveletTotalEntropy(X,'db2')
	WTentrbio = extractWaveletTotalEntropy(X,'rbio3.3')
	WTentsym = extractWaveletTotalEntropy(X,'sym3')
	Wentdb = extractWaveletEntropy2(X,window,'db20')
	Wentrbio = extractWaveletEntropy2(X,window,'rbio3.3')
	WT2entrbio = extractWaveletTotalEntropy2(X,'rbio3.3')
	WT2entsym = extractWaveletTotalEntropy2(X,'sym3')
	cwtb = freqNotation(X)

	return np.concatenate([Energy,[MYOP,WAMP,ZC,MFL,MAV,VAR,Went,WTentdb,WTentrbio,WTentsym],Wentdb,Wentrbio,WT2entrbio,WT2entsym,cwtb])

def addWaveletInfoV3(X,level,window):
 	res,MYOP,WAMP,ZC,MFL,MAV,VAR = extractWaveletV2(X,level)
 	res1 = extractWaveletEntropy(X,window,'db2')
 	res2 = extractWaveletTotalEntropy(X,'db2')
 	res3 = extractWaveletTotalEntropy(X,'rbio3.3')
 	res4 = extractWaveletTotalEntropy(X,'sym3')
 	res5 = extractWaveletEntropy2(X,window,'db2')
 	res6 = extractWaveletEntropy2(X,window,'rbio3.3')
 	res7 = extractWaveletTotalEntropy2(X,'rbio3.3')
 	res8 = extractWaveletTotalEntropy2(X,'sym3')
 	res9 = freqNotation(X)

 	return np.concatenate([res,[MYOP,WAMP,ZC,MFL,MAV,VAR,res1,res2,res3,res4],res5,res6,res7,res8,res9])

def addGlobalInfo(X):
	fish = fisher_info(X,3,10)
	svd = svd_entropy(X,1,20)
	spec = spectral_entropy(X,200,np.linspace(0.5,84.5,22))
	kurt = stats.kurtosis(X)
	skew = stats.skew(X)
	#res7 =  stats.kstat(X)
	#res8 =  stats.moment(X,2)
	#res9 =  stats.moment(X,3)
	aac = averageAmplitudeChange(X)
	dasd = differenceAbsoluteSTD(X)
	peak = peaks(X)
	zero = zeros(X)
	return [fish,svd,spec,kurt,skew,aac,dasd,peak,zero]

def addFractalInfo(X):
	hur = hurst(X)
	pf = pfd(X)
	hf = hfd(X,8)
	df = dfa(X)
	res5,res6,res7 = hjorth(X)
	return [hur,pf,hf,df,res5,res6,res7]

def addFilteredInfo(X):
	apen = ap_entropy(X,2,1.5)
	aac = averageAmplitudeChange(X)
	dasd = differenceAbsoluteSTD(X)
	std = np.std(X)
	kol = extractKolmogorov(X)
	#samp1 = samp_entropy(X,2,1.5,1)
	#samp2 = samp_entropy(X,2,1.5,2)
	#samp3 = samp_entropy(X,2,1.5,3)
	#samp4 = samp_entropy(X,2,1.5,4)
	return [apen,aac,dasd,std,kol]




