from rpy2.robjects.packages import importr
from rpy2 import robjects
import numpy as np

C50 = importr('C50')    
C5_0 = robjects.r('C5.0')
predict = robjects.r('predict')

def testTree_CNet(test_data,decisionTree):
  
    count = 0 # Number of correct predictions 
  
    # Predict new data
    if (decisionTree.size > 0):
        prediction = predict(decisionTree, newdata = test_data[:,:-1])
        prediction = np.array(prediction)
    
        for i in range(len(prediction)):
            if (prediction[i] == test_data[i,-1]):
                count += 1
              
        # return value
        accuracy = count/len(test_data)
    
    else:
        accuracy = 0.5
      
    return accuracy 
   


