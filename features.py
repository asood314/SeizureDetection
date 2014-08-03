import numpy as np
import pandas as pd
#functions for calculating features for use in seizure detection algorithm

def allFeatures(data):
    output = []
    chans = len(data.ix[1,:])
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    maxFreqs = []
    times = len(data.ix[:,1])-1 
    delta = data.values[1:,:] - data.values[:times,:]
    glob = pd.DataFrame(delta)
    freqGlob = np.fft.fftfreq(len(glob.ix[:,0]),1.0/len(glob.ix[:,0]))
    maxFreqsGlob = []
    featureList = []
# Channel Features
    for i in range(1,chans):
        output.append(data.ix[:,i].abs().max())
        featureList.append('chan%iMaxAmp'%(i-1))
        output.append(data.ix[:,i].abs().mean())
        featureList.append('chan%iMeanAmp'%(i-1))
        output.append(data.ix[:,i].abs().var()) 
        featureList.append('chan%iVarAbs'%(i-1))   
        output.append(data.ix[:,i].var())
        featureList.append('chan%iVar'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).max())
        featureList.append('chan%iMaxFourAmp'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).mean())
        featureList.append('chan%iMeanFourAmp'%(i-1))
        output.append(abs(np.fft.fft(data.ix[:,i])).var())
        featureList.append('chan%iVarFourAmp'%(i-1))
        ft = abs(np.fft.fft(data.ix[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
        featureList.append('chan%iMaxFreq'%(i-1))
# Derivative Channel Features
        output.append(np.max(np.abs(delta[:,i])))
        featureList.append('chan%iMaxDel'%(i-1))
        output.append(np.mean(np.abs(delta[:,i])))
        featureList.append('chan%iMeanDel'%(i-1))
        output.append(np.var(np.abs(delta[:,i])))
        featureList.append('chan%iVarAbsDel'%(i-1))
        output.append(np.var(delta[:,i]))
        featureList.append('chan%iVarDel'%(i-1))
        output.append(np.max(np.abs(np.fft.fft(delta[:,i]))))
        featureList.append('chan%iMaxDelFour'%(i-1))
        output.append(np.mean(np.abs(np.fft.fft(delta[:,i]))))
        featureList.append('chan%iMeanDelFour'%(i-1))
        output.append(np.var(np.abs(np.fft.fft(delta[:,i]))))
        featureList.append('chan%iVarDelFour'%(i-1))
        ft = np.abs(np.fft.fft(delta[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
        featureList.append('chan%iMaxDelFreq'%(i-1))
# Global Features
    output.append(data.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxAmp')
    output.append(data.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanAmp')
    output.append(data.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varAmpAbs')
    output.append(data.ix[:,1:].apply(np.max).var())
    featureList.append('varAmp')
    output.append(data.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMean')
    output.append(data.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVar')
    output.append(data.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVar')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).max())
    featureList.append('maxFourAmp')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).mean())
    featureList.append('meanFourAmp')
    output.append(np.array([abs(np.fft.fft(data['chan%i'%i])).max() for i in range(len(data.columns) - 1)]).var())
    featureList.append('varFourAmp')
    for i in range(len(data.columns)-1):
        ft = abs(np.fft.fft(data['chan%i'%i]))
        maxFreqs.append(abs(freq[np.argmax(ft)]))
    output.append(np.max(maxFreqs))
    featureList.append('maxFreq')
    output.append(np.mean(maxFreqs))
    featureList.append('meanFreq')
    output.append(np.var(maxFreqs))
    featureList.append('varFreq')
# Derivative Global Features
    output.append(glob.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxDel')
    output.append(glob.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanDel')
    output.append(glob.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varDelAbs')
    output.append(glob.ix[:,1:].apply(np.max).var())
    featureList.append('varDel')
    output.append(glob.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMeanDel')
    output.append(glob.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVarDel')
    output.append(glob.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVarDel')
    output.append(np.array([abs(np.fft.fft(glob.ix[:,i])).max() for i in range(len(glob.columns) - 1)]).max())
    featureList.append('maxFourDel')
    output.append(np.array([abs(np.fft.fft(glob.ix[:,i])).max() for i in range(len(glob.columns) - 1)]).mean())
    featureList.append('meanFourDel')
    output.append(np.array([abs(np.fft.fft(glob.ix[:,i])).max() for i in range(len(glob.columns) - 1)]).var())
    featureList.append('varFourDel')
    for i in range(len(glob.columns)-1):
        ft = abs(np.fft.fft(glob.ix[:,i]))
        maxFreqsGlob.append(abs(freqGlob[np.argmax(ft)]))
    output.append(np.max(maxFreqsGlob))
    featureList.append('maxFreqDel')
    output.append(np.mean(maxFreqsGlob))
    featureList.append('meanFreqDel')
    output.append(np.var(maxFreqsGlob))
    featureList.append('varFreqDel')
    return pd.DataFrame({'allFeats':output},index=featureList).T

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
funcDict = {'allFeats'      : allFeatures,
            'maxAmp'       : maximumAmplitude,
            'meanAmp'      : meanAmplitude,
            'varAmp'       : varianceOfAmplitude,
            'varMean'      : varianceOfMean,
            'varVar'       : varianceOfVariance,
            'meanVar'      : meanOfVariance,
            'maxFourAmp'   : maxFourierAmplitude,
            'meanFourAmp'  : meanFourierAmplitude,
            'varFourAmp'   : varianceOfFourierAmplitude,
            'maxFreq'      : maxFrequency,
            'meanFreq'     : meanOfMaxFrequency,
            'varFreq'      : varianceOfMaxFrequency}
