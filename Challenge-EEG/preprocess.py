from numpy import sin, arange, pi
from scipy.signal import lfilter, firwin, butter
from pylab import figure, plot, grid, show
import numpy as np
import pywt as pywt

def waveletpreprocess(X,threshold=20,threshold2=40):
	wave_coefs = pywt.wavedec(X,'haar',mode='sym',level=10)
	for coefs in wave_coefs[1:]:
		coefs[np.abs(coefs)<=threshold]=0
		bcast = [coef>threshold and coef<=threshold2 for coef in np.abs(coefs)]
		coefs[bcast] = np.sign(coefs[bcast])*(threshold2*(np.abs(coefs[bcast])-threshold)/(threshold2-threshold))
	filtered_signal=pywt.waverec(wave_coefs,'haar')
	return filtered_signal


def waveletpre(X,level):
	c = pywt.wavedec(X,'db7',mode='sym',level=level)
	a = pywt.idwt(c[0],None,'db7')
	return a


def preprocessSignal(X,plotting=False):

	signal=X
	#------------------------------------------------
	# Signal Information
	#------------------------------------------------
	# 6000 samples at 200 Hz
	sample_rate = 200

	nsamples = 6000
	 
	t=np.linspace(0,30,6000)


	#------------------------------------------------
	# Create a FIR filter and apply it to signal.
	#------------------------------------------------
	# The Nyquist rate of the signal.
	nyq_rate = sample_rate / 2.
	 
	# The cutoff frequency of the filter: 6KHz
	cutoff_hz = 0.5
	 
	# Length of the filter (number of coefficients, i.e. the filter order + 1)
	numtaps = 29
	 
	# Use firwin to create a lowpass FIR filter
	fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate)
	 
	# Use lfilter to filter the signal with the FIR filter
	#filtered_signal = lfilter(fir_coeff, 1.0, signal)



	#------------------------------------------------
	# Create a FIR filter Bandpass
	#------------------------------------------------
	# The Nyquist rate of the signal.
	nyq_rate = sample_rate / 2.
	 
	# The cutoff frequency of the filter: 0.4 KHz
	cutoff_hz = np.array([])
	 
	def butter_bandpass(lowcut, highcut, fs, order=5):
	    nyq = 0.5 * fs
	    low = lowcut / nyq
	    high = highcut / nyq
	    b, a = butter(order, [low, high], btype='band')
	    return b, a
	
	def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	    y = lfilter(b, a, data)
	    return y
	 
	# Use lfilter to filter the signal with the FIR filter
	filtered_signal = butter_bandpass_filter(X,0.5,45,200,4)


	def dampening(X,a,b):
	    res=np.zeros(6000)
	    for i in range(6000):
	    	res[i]=np.sign(X[i])*np.sqrt((a*X[i]*np.cos(float(i)/200))**2 + (b*X[i]*np.sin(float(i)/200))**2)
	    return res

	filtered_dampened_signal = dampening(filtered_signal,0.75,0.75)

	if plotting:
		#------------------------------------------------
		# Plot the original and filtered signals.
		#------------------------------------------------
		 
		# The first N-1 samples are "corrupted" by the initial conditions
		#warmup = numtaps - 1
		 
		# The phase delay of the filtered signal
		#delay = (warmup / 2) / sample_rate
		 
		figure(1)
		# Plot the original signal
		plot(t, signal)
		 
		# Plot the filtered signal, shifted to compensate for the phase delay
		plot(t, filtered_signal, 'r-')
		plot(t, filtered_dampened_signal,'g-')
		 
		#plot(t-delay, filtered_signal2, 'r-')
		# Plot just the "good" part of the filtered signal.  The first N-1
		# samples are "corrupted" by the initial conditions.
		#plot(t[warmup:]-delay, filtered_signal[warmup:], 'g', linewidth=4)
		 
		grid(True)
		 
		show()

	return filtered_dampened_signal