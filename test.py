from utils import *

#select all dogs/patients, don't set aside any data for validation
dataSelector = [['Dog_1',0,0],
                ['Dog_2',0,0],
                ['Dog_3',0,0],
                ['Dog_4',0,0],
                ['Patient_1',0,0],
                ['Patient_2',0,0],
                ['Patient_3',0,0],
                ['Patient_4',0,0],
                ['Patient_5',0,0],
                ['Patient_6',0,0],
                ['Patient_7',0,0],
                ['Patient_8',0,0]]
print "Loading train/validation samples using selector:\n",dataSelector
samples = loadTrainAndValidationSamples(dataSelector,['allFeats'],100.0)
print "Training sample size: ",samples['train'].shape
forest = trainDoubleForest(samples['train'])
print "Done training. Loading test samples..."

testSam = loadTestSample(['allFeats'],100.0)
print "Test sample size: ",testSam.shape
makeSubmission([forest['seizure'],forest['early']],testSam)
print "Done."

