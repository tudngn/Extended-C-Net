from rpy2.robjects.packages import importr
from rpy2 import robjects
import pickle
from testTree_CNet import testTree_CNet

C50 = importr('C50')    
C5_0 = robjects.r('C5.0')
C5_0Control = robjects.r('C5.0Control')

def buildTree_CNet(filename,trainingData,testingData):
  
    Directory = "./balance/C5/C-Net/"
    # build tree
    decisionTree = C50.C5_0(x = trainingData[:,:-1], 
                            y = robjects.vectors.FactorVector(trainingData[:,-1]),
                            control = C5_0Control(noGlobalPruning = True),
                            rules = True)
    tree_filename = Directory + filename + "_Tree.pkl"
    with open(tree_filename, 'wb') as f0:
        pickle.dump(decisionTree, f0)                  

    # build pruned tree
    prunedTree = C50.C5_0(x = trainingData[:,:-1], 
                            y = robjects.vectors.FactorVector(trainingData[:,-1]),
                            control = C5_0Control(noGlobalPruning = False),
                            rules = True)
    prunedtree_filename = Directory + filename + "_PrunedTree.pkl"
    with open(prunedtree_filename, 'wb') as f1:
        pickle.dump(prunedTree, f1)                  

    # Predict test data
    accuracies = testTree_CNet(testingData,decisionTree)
    pruned_accuracies = testTree_CNet(testingData,prunedTree)
    tree_size = max(decisionTree.size,1)
    pruned_tree_size = max(prunedTree.size,1)

    return(accuracies, pruned_accuracies, tree_size, pruned_tree_size)
  
