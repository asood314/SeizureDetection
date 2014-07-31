import numpy as np
import pandas as pd
#functions for calculating features for use in seizure detection algorithm

def allFeatures(data):
    output = []
    chans = len(data.ix[1,:])
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    maxFreqs = []
# Channel Features
    for i in range(1,chans):
        output.append(data.ix[:,i].abs().max())
    for i in range(1,chans):
        output.append(data.ix[:,i].abs().mean())
    for i in range(1,chans):
        output.append(data.ix[:,i].abs().var())    
    for i in range(1,chans):
        output.append(data.ix[:,i].var())
    for i in range(1,chans):
        output.append(abs(np.fft.fft(data.ix[:,i])).max())
    for i in range(1,chans):
        output.append(abs(np.fft.fft(data.ix[:,i])).mean())
    for i in range(1,chans):
        output.append(abs(np.fft.fft(data.ix[:,i])).var())
    for i in range(1,chans):
        ft = abs(np.fft.fft(data.ix[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
# Global Features
    output.append(data.ix[:,1:].abs().apply(np.max).max())
    output.append(data.ix[:,1:].abs().apply(np.max).mean())
    output.append(data.ix[:,1:].abs().apply(np.max).var())
    output.append(data.ix[:,1:].apply(np.max).var())
    output.append(data.ix[:,1:].abs().apply(np.mean).var())
    output.append(data.ix[:,1:].abs().apply(np.var).var())
    output.append(data.ix[:,1:].abs().apply(np.var).mean())
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).max())
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).mean())
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).var())
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    output.append(np.max(maxFreqs))
    output.append(np.mean(maxFreqs))
    output.append(np.var(maxFreqs))
    return pd.DataFrame({'allFeats':output}).T

def maximumAmplitude(data):
    #maximum unsigned reading accross all channels
    return pd.DataFrame({'maxAmp':[data.ix[:,1:].abs().apply(np.max).max()]})
                        
def meanAmplitude(data):
    #mean of the maximum unsigned reading in each channel
    return pd.DataFrame({'meanAmp':[data.ix[:,1:].abs().apply(np.max).mean()]})

def varianceOfAmplitude(data):
    #variance of the maximum unsigned reading in each channel
    return pd.DataFrame({'varAmp':[data.ix[:,1:].abs().apply(np.max).var()]})

def varianceOfMean(data):
    #variance of the mean reading in each channel
    return pd.DataFrame({'varMean':[data.ix[:,1:].abs().apply(np.mean).var()]})

def varianceOfVariance(data):
    #variance of the variance of the readings in each channel
    return pd.DataFrame({'varVar':[data.ix[:,1:].abs().apply(np.var).var()]})

def meanOfVariance(data):
    #mean of the variance of the readings in each channel
    return pd.DataFrame({'meanVar':[data.ix[:,1:].abs().apply(np.var).mean()]})
    
    

def maxFourierAmplitude(data):
    #maximum of unsigned height of the peak in fourier space for each channel
    return pd.DataFrame({'maxFourAmp':[np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).max()]})

def meanFourierAmplitude(data):
    #maximum of unsigned height of the peak in fourier space for each channel
    return pd.DataFrame({'meanFourAmp':[np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).mean()]})

def varianceOfFourierAmplitude(data):
    #maximum of unsigned height of the peak in fourier space for each channel
    return pd.DataFrame({'varFourAmp':[np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).var()]})

def maxFrequency(data):
    #maximum of frequency of largest peak in each channel
    maxFreqs = []
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    return pd.DataFrame({'maxFreq':[np.max(maxFreqs)]})

def meanOfMaxFrequency(data):
    #maximum of frequency of largest peak in each channel
    maxFreqs = []
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    return pd.DataFrame({'meanFreq':[np.mean(maxFreqs)]})

def varianceOfMaxFrequency(data):
    #maximum of frequency of largest peak in each channel
    maxFreqs = []
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    return pd.DataFrame({'varFreq':[np.var(maxFreqs)]})

#Dictionary associates functions with string names so that features can be easily selected later
funcDict = {'allFeats'      : allFeatures}#,
            #'maxAmp'       : maximumAmplitude,
            #'meanAmp'      : meanAmplitude,
            #'varAmp'       : varianceOfAmplitude,
            #'varMean'      : varianceOfMean,
            #'varVar'       : varianceOfVariance,
            #'meanVar'      : meanOfVariance,
            #'maxFourAmp'   : maxFourierAmplitude,
            #'meanFourAmp'  : meanFourierAmplitude,
            #'varFourAmp'   : varianceOfFourierAmplitude,
            #'maxFreq'      : maxFrequency,
            #'meanFreq'     : meanOfMaxFrequency,
            #'varFreq'      : varianceOfMaxFrequency}
