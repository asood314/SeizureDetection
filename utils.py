import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import csv
from features import *

dataDirectory = "data/clips"

def setDataDirectory(dirName):
    global dataDirectory
    dataDirectory = dirName
    return

def loadData(matlabFile,lat=0):
    matlabDict = scipy.io.loadmat(matlabFile)
    if matlabFile.count('_ictal_') > 0:
        lat = matlabDict['latency'][0]
    freq = len(matlabDict['data'][0])
    data = pd.DataFrame({'time':np.arange(lat,1.0+lat,1.0/freq)})
    for i in range(len(matlabDict['channels'][0][0])):
        channelName = "chan%i"%i
        data[channelName] = matlabDict['data'][i]
    return data

def downSample(data,factor):
      coarseData = data.groupby(lambda x: int(np.floor(x/factor))).mean()
      return coarseData

def plotChannels(data,channels,plotOpts):
    if len(channels) > len(plotOpts):
        print 'ERROR: Must specify plot options for each channel'
        return
    for chan in range(len(channels)):
        plt.plot(data['time'],data[channels[chan]],plotOpts[chan])
        plt.xlabel('time (s)')
        plt.ylabel('Electrode reading')
        plt.legend(channels)

def loadClips(patient,clipType,clipNumbers,targetFrequency):
    data = []
    for clip in clipNumbers:
        clipPath = "%s/%s/%s_%s_segment_%i.mat"%(dataDirectory,patient,patient,clipType,clip)
        tmpData = loadData(clipPath,clip-1)
        downFactor = float(len(tmpData['time'])) / targetFrequency
        if downFactor > 1.0:
            data.append(downSample(tmpData,downFactor))
        else:
            data.append(tmpData)
    return pd.concat(data)

def convertToFeatureSeries(data,featureFunctions,isSeizure=False,latency=0,isTest=False,testFile=""):
    #converts time series data into a set of features
    #featureFunctions should be a list of the desired features, which must be defined in funcDict
    #isSeizure and latency are used to add that information for training/validation
    #when loading test samples, isTest should be set True and the file name specified so that this information is available when writing the submission file
    global funcDict
    data['time'] = data['time'] - latency
    features = []
    for func in featureFunctions:
        features.append(funcDict[func](data))
    data = pd.concat(features,axis=1)
    if not isTest:
        data['latency'] = latency
        data['isSeizure'] = int(isSeizure)
        data['isEarly'] = int(latency < 19 and isSeizure)
    else:
        data['testFile'] = testFile
    return data

def loadTrainAndValidationSamples(dataSelector,featureFunctions,commonFrequency=-1):
    #loads training samples and optionally splits off a chunk for validation
    #dataSelector is a list of lists that have the form [patientName,fraction of seizure segments to use for validation,fraction of non-seizure segments for validation
    #--for example [['Patient_2',0.5,0.5],['Dog_3',0.2,0.2]] would load the data for patient 2 and dog 3, putting half of the patient 2 data and 20% of the dog 3 data in the validation sample and the rest in the training sample
    #featureFunctions specifies the list of features to use
    #commonFrequency option is used to downsample the data to that frequency
    entriesTrain = []
    entriesValid = []
    for patient in dataSelector:
        files = os.listdir('%s/%s'%(dataDirectory,patient[0]))
        ictal = []
        interictal = []
        for phil in files:
            if phil.count('_inter') > 0:
                interictal.append(phil)
            elif phil.count('_ictal_') > 0:
                ictal.append(phil)
        for i in ictal:
            tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],i))
            lat = tmpData['time'][0]
            if commonFrequency > 0:
                downFactor = float(len(tmpData['time'])) / commonFrequency
                if downFactor > 1.0:
                    tmpData = downSample(tmpData,downFactor)
            featureSet = convertToFeatureSeries(tmpData,featureFunctions,True,lat)
            if np.random.random() > patient[1]:
                entriesTrain.append(featureSet)
            else:
                entriesValid.append(featureSet)
        for ii in interictal:
            tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],ii))
            lat = tmpData['time'][0]
            if commonFrequency > 0:
                downFactor = float(len(tmpData['time'])) / commonFrequency
                if downFactor > 1.0:
                    tmpData = downSample(tmpData,downFactor)
            featureSet = convertToFeatureSeries(tmpData,featureFunctions,False,0)
            if np.random.random() > patient[2]:
                entriesTrain.append(featureSet)
            else:
                entriesValid.append(featureSet)
    if len(entriesTrain) == 0:
        print "ERROR: No entries in training sample"
        return {'train':0,'validation':0}
    trainSample = pd.concat(entriesTrain,ignore_index=True)
    if len(entriesValid) == 0:
        return {'train':trainSample,'validation':0}
    validSample = pd.concat(entriesValid,ignore_index=True)
    return {'train':trainSample,'validation':validSample}

def loadTestSample(featureFunctions,commonFrequency=-1):
    #loads test data
    #arguments same as corresponding arguments for loadTrainAndValidationSamples
    patientList = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']
    entries = []
    for patient in patientList:
        files = os.listdir('%s/%s'%(dataDirectory,patient))
        for phil in files:
            if phil.count('test') > 0:
                tmpData = loadData("%s/%s/%s"%(dataDirectory,patient,phil))
                if commonFrequency > 0:
                    downFactor = float(len(tmpData['time'])) / commonFrequency
                    if downFactor > 1.0:
                        tmpData = downSample(tmpData,downFactor)
                featureSet = convertToFeatureSeries(tmpData,featureFunctions,isTest=True,testFile=phil)
                entries.append(featureSet)
    testSample = pd.concat(entries,ignore_index=True)
    return testSample
    
def loadIndivTestSamples(dataSelector, featureFunctions,commonFrequency=-1):
    #loads test data
    #arguments same as corresponding arguments for loadTrainAndValidationSamples
    #patientList = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']
    entries = []
    for patient in dataSelector:
        files = os.listdir('%s/%s'%(dataDirectory,patient[0]))
        for phil in files:
            if phil.count('test') > 0:
                tmpData = loadData("%s/%s/%s"%(dataDirectory,patient[0],phil))
                if commonFrequency > 0:
                    downFactor = float(len(tmpData['time'])) / commonFrequency
                    if downFactor > 1.0:
                        tmpData = downSample(tmpData,downFactor)
                featureSet = convertToFeatureSeries(tmpData,featureFunctions,isTest=True,testFile=phil)
                entries.append(featureSet)
    testSample = pd.concat(entries,ignore_index=True)
    return testSample

def trainRandomForest(trainDF):
    #trains a random forest on the training sample and returns the trained forest
    trainArray = trainDF.values
    forest = RandomForestClassifier(n_estimators=1000)
    return forest.fit(trainArray[:,0:-3],trainArray[:,-2:])

def validateRandomForest(forest,validDF,latencyBinWidth=-1):
    #prints efficiency and false positive metrics and plots efficiency vs. latency for a given forest using the validation sample
    output = forest.predict(validDF.values[:,0:-3])
    validDF['PiS'] = output[:,0].astype(int)
    validDF['PiE'] = output[:,1].astype(int)
    for key,group in validDF.groupby('isSeizure'):
        if key:
            print "Efficiency for seizure detection: ",group['PiS'].mean()
            for k,g in group.groupby('isEarly'):
                if k:
                    print "Efficiency for early seizure detection: ",g['PiE'].mean()
            df = group.groupby('latency').mean()
            if latencyBinWidth > 1.0:
                df = downSample(df,latencyBinWidth)
            plt.plot(np.array(df.index),df['PiS'],'b-')
            plt.plot(np.array(df.index),df['PiE'],'r-')
            plt.xlabel('latency')
            plt.ylabel('efficiency')
            plt.title('Detection efficiency vs. Latency')
            plt.savefig('efficiencySeizure.png')
        else:
            print "False positive rate for seizure: ",group['PiS'].mean()
            print "False positive rate for early seizure: ",group['PiE'].mean()
    return validDF

def trainDoubleForest(trainDF):
    #trains a random forest on the training sample and returns the trained forest
    trainArray = trainDF.values
    forestSeizure = ExtraTreesClassifier(n_estimators=1000, min_samples_split = 1)
    forestEarly = ExtraTreesClassifier(n_estimators=1000, min_samples_split = 1)
    return {'seizure':forestSeizure.fit(trainArray[:,0:-3],trainArray[:,-2]),'early':forestEarly.fit(trainArray[:,0:-3],trainArray[:,-1])}

def validateDoubleForest(forests,validDF,latencyBinWidth=-1):
    #prints efficiency and false positive metrics and plots efficiency vs. latency for a given forest using the validation sample
    seizure = forests['seizure'].predict(validDF.values[:,0:-3])
    early = forests['early'].predict(validDF.values[:,0:-3])
    validDF['PiS'] = seizure.astype(int)
    validDF['PiE'] = early.astype(int)
    for key,group in validDF.groupby('isSeizure'):
        if key:
            print "Efficiency for seizure detection: ",group['PiS'].mean()
            for k,g in group.groupby('isEarly'):
                if k:
                    print "Efficiency for early seizure detection: ",g['PiE'].mean()
            df = group.groupby('latency').mean()
            if latencyBinWidth > 1.0:
                df = downSample(df,latencyBinWidth)
            plt.plot(np.array(df.index),df['PiS'],'b-')
            plt.plot(np.array(df.index),df['PiE'],'r-')
            plt.xlabel('latency')
            plt.ylabel('efficiency')
            plt.title('Detection efficiency vs. Latency')
            plt.savefig('efficiencySeizure.png')
        else:
            print "False positive rate for seizure: ",group['PiS'].mean()
            print "False positive rate for early seizure: ",group['PiE'].mean()
    return validDF

def testProbs(forestList,testDF):
    #runs the forest on the test sample and returns output
    output = []
    for forest in forestList:
        output.append(forest.predict_proba(testDF.values[:,0:-1])[:,1])
    output = np.array(output).T
    return output 
    
def makeSubmit(output, testDF):
    # writes submission file
    outFile = open("submission.csv","wb")
    csv_writer = csv.writer(outFile)
    csv_writer.writerow(['clip','seizure','early'])
    csv_writer.writerows(zip(testDF['testFile'].values,output[:,0].astype(float),output[:,1].astype(float)))
    outFile.close()
    return

def makeSubmission(forestList,testDF):
    #runs the forest on the test sample and writes submission file
    output = []
    for forest in forestList:
        output.append(forest.predict_proba(testDF.values[:,0:-1])[:,1])
    output = np.array(output).T
    outFile = open("submission.csv","wb")
    csv_writer = csv.writer(outFile)
    csv_writer.writerow(['clip','seizure','early'])
    csv_writer.writerows(zip(testDF['testFile'].values,output[:,0].astype(float),output[:,1].astype(float)))
    outFile.close()
    return

def plotFeatures(feat1,feat2,data):
    m1 = np.mean(data[feat1])
    s1 = np.std(data[feat1])
    m2 = np.mean(data[feat2])
    s2 = np.std(data[feat2])
    plt.subplot(1,2,1)
    for key,group in data.groupby('isSeizure'):
        if key:
            plt.plot((group[feat1]-m1)/s1,(group[feat2]-m2)/s2,'ro')
        else:
            plt.plot((group[feat1]-m1)/s1,(group[feat2]-m2)/s2,'bo')
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend(( 'seizure' , 'non-seizure' ))
    plt.subplot(1,2,2)
    for key,group in data.groupby('isEarly'):
        if key:
            plt.plot((group[feat1]-m1)/s1,(group[feat2]-m2)/s2,'ro')
        else:
            plt.plot((group[feat1]-m1)/s1,(group[feat2]-m2)/s2,'bo')
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend(( 'early' , 'non-early' ))
    plt.savefig('%s_v_%s.png'%(feat2,feat1))
