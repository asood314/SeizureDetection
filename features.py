import numpy as np
import pandas as pd
import itertools
#functions for calculating features for use in seizure detection algorithm

def allFeatures(data):
    output = []
    chans = len(data.ix[1,:])
    freq = np.fft.fftfreq(len(data['time']),1.0/len(data['time']))
    maxFreqs = []
    times = len(data.ix[:,1])-1 
    featureList = []
    delta1 = data.values[1:,:] - data.values[:times,:] # the 1st derivative
    glob1 = pd.DataFrame(delta1)
    freqGlob1 = np.fft.fftfreq(len(glob1.ix[:,0]),1.0/len(glob1.ix[:,0]))
    maxFreqsGlob1 = []
    delta2 = delta1[1:,:] - delta1[:times-1,:] # the 2nd derivative
    glob2 = pd.DataFrame(delta2)
    freqGlob2 = np.fft.fftfreq(len(glob2.ix[:,0]),1.0/len(glob2.ix[:,0]))
    maxFreqsGlob2 = []
    combs = []
    [combs.append(a) for a in (itertools.combinations(list(data.columns)[1:], 2))]
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
# 1st Derivative Channel Features
        output.append(np.max(np.abs(delta1[:,i])))
        featureList.append('chan%iMaxDel1'%(i-1))
        output.append(np.mean(np.abs(delta1[:,i])))
        featureList.append('chan%iMeanDel1'%(i-1))
        output.append(np.var(np.abs(delta1[:,i])))
        featureList.append('chan%iVarAbsDel1'%(i-1))
        output.append(np.var(delta1[:,i]))
        featureList.append('chan%iVarDel1'%(i-1))
        output.append(np.max(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iMaxDel1Four'%(i-1))
        output.append(np.mean(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iMeanDel1Four'%(i-1))
        output.append(np.var(np.abs(np.fft.fft(delta1[:,i]))))
        featureList.append('chan%iVarDel1Four'%(i-1))
        ft = np.abs(np.fft.fft(delta1[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
        featureList.append('chan%iMaxDel1Freq'%(i-1))
# 2nd Derivative Channel Features
        output.append(np.max(np.abs(delta2[:,i])))
        featureList.append('chan%iMaxDel2'%(i-1))
        output.append(np.mean(np.abs(delta2[:,i])))
        featureList.append('chan%iMeanDel2'%(i-1))
        output.append(np.var(np.abs(delta2[:,i])))
        featureList.append('chan%iVarAbsDel2'%(i-1))
        output.append(np.var(delta2[:,i]))
        featureList.append('chan%iVarDel2'%(i-1))
        output.append(np.max(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iMaxDel2Four'%(i-1))
        output.append(np.mean(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iMeanDel2Four'%(i-1))
        output.append(np.var(np.abs(np.fft.fft(delta2[:,i]))))
        featureList.append('chan%iVarDel2Four'%(i-1))
        ft = np.abs(np.fft.fft(delta2[:,i]))
        output.append(abs(freq[np.argmax(ft)]))
        featureList.append('chan%iMaxDel2Freq'%(i-1))
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
# 1st Derivative Global Features
    output.append(glob1.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varDel1Abs')
    output.append(glob1.ix[:,1:].apply(np.max).var())
    featureList.append('varDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMeanDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVarDel1')
    output.append(glob1.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVarDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).max())
    featureList.append('maxFourDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).mean())
    featureList.append('meanFourDel1')
    output.append(np.array([abs(np.fft.fft(glob1.ix[:,i])).max() for i in range(len(glob1.columns) - 1)]).var())
    featureList.append('varFourDel1')
    for i in range(len(glob1.columns)-1):
        ft = abs(np.fft.fft(glob1.ix[:,i]))
        maxFreqsGlob1.append(abs(freqGlob1[np.argmax(ft)]))
    output.append(np.max(maxFreqsGlob1))
    featureList.append('maxFreqDel1')
    output.append(np.mean(maxFreqsGlob1))
    featureList.append('meanFreqDel1')
    output.append(np.var(maxFreqsGlob1))
    featureList.append('varFreqDel1')
# 2nd Derivative Global Features
    output.append(glob2.ix[:,1:].abs().apply(np.max).max())
    featureList.append('maxDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.max).mean())
    featureList.append('meanDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.max).var())
    featureList.append('varDelAbs2')
    output.append(glob2.ix[:,1:].apply(np.max).var())
    featureList.append('varDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.mean).var())
    featureList.append('varMeanDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.var).var())
    featureList.append('varVarDel2')
    output.append(glob2.ix[:,1:].abs().apply(np.var).mean())
    featureList.append('meanVarDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).max())
    featureList.append('maxFourDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).mean())
    featureList.append('meanFourDel2')
    output.append(np.array([abs(np.fft.fft(glob2.ix[:,i])).max() for i in range(len(glob2.columns) - 1)]).var())
    featureList.append('varFourDel2')
    for i in range(len(glob2.columns)-1):
        ft = abs(np.fft.fft(glob2.ix[:,i]))
        maxFreqsGlob2.append(abs(freqGlob2[np.argmax(ft)]))
    output.append(np.max(maxFreqsGlob2))
    featureList.append('maxFreqDel2')
    output.append(np.mean(maxFreqsGlob2))
    featureList.append('meanFreqDel2')
    output.append(np.var(maxFreqsGlob2))
    featureList.append('varFreqDel2')
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
    
def covariances(data):
    # inter-channel covariance
    combs = []
    # combinations of all channels 
    [combs.append(a) for a in (itertools.combinations(list(data.columns)[1:], 2))] 
    covs = [np.cov(data.ix[:,combs[k][0]], data.ix[:,combs[k][1]])[0,1] for k in np.arange(0,len(combs)) ]
    return pd.DataFrame({'coVar':[np.mean(np.abs(covs))]})

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
            'varFreq'      : varianceOfMaxFrequency,
            'coVar'        : covariances
            }
