from utils import *
from sklearn.metrics import roc_auc_score

def validationProbs(forestList,testDF):
    output = []
    for forest in forestList:
        output.append(forest.predict_proba(testDF.values[:,0:-3])[:,1])
    output = np.array(output).T
    return output

dataSelector = [['Patient_8',0.5,0.5]]
                
predictions = []; validations = []; testSamples = pd.DataFrame()
                
for num, dataSet in enumerate(dataSelector):
    #print dataSet, num + 1
    print "Loading train/validation samples using selector:\n",dataSelector[num] 
    print "(lat, freq, trees) = ", (18, 100,1000)
    samples = loadTrainAndValidationSamples([dataSet],['allFeats'],100.0)
    print "Training sample size: ",samples['train'].shape
    print "Validation sample size: ",samples['validation'].shape
    print "Training..."
    forest = trainDoubleForest(samples['train'])
    print "Done training. Making Predictions..."

    testSam = samples['validation'] 
    testSamples = pd.concat([testSamples, testSam])    
    #print "Test sample size: ",testSam.shape
    
    col = len(testSam.columns)
    row = len(testSam)
    predictions = validationProbs([forest['seizure'],forest['early']],testSam)
    validations = testSam.values[:, col-2:col]
    for i in range(0,row):
        if testSam.values[i,-1] > 0 and testSam.values[i,-3] > 15:
           validations[i,1] = 0 
    #above is needed to ensure that validations use the correct lat < 16       
    
    seizureAUC = roc_auc_score(validations[:,0], predictions[:,0])
    earlyAUC = roc_auc_score(validations[:,1], predictions[:,1])
    averageAUC = (0.5)*(seizureAUC + earlyAUC)
    
    
    print "Seizure AUC: ", seizureAUC
    print "Early AUC: ", earlyAUC
    print "Average AUC: ", averageAUC
   
#makeSubmit(np.array(predictions), testSamples)

#print "Done."

