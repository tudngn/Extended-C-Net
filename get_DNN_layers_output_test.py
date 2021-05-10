# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:52:39 2019

@author: z5095790
"""

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from scipy.special import expit
from sklearn.utils.extmath import softmax

def load_weight_bias(Model):

    DNNModel = Model
    
    weight1 = DNNModel.layers[0].get_weights()[0]	
    biase1 = DNNModel.layers[0].get_weights()[1] #2,4
    
    weight2 = DNNModel.layers[2].get_weights()[0]
    biase2 = DNNModel.layers[2].get_weights()[1] #2,4
    
    weight3 = DNNModel.layers[4].get_weights()[0]
    biase3 = DNNModel.layers[4].get_weights()[1] #2,4
    
    weight = [weight1, weight2, weight3]
    bias = [biase1, biase2, biase3]
    
    return(weight, bias)
    
    
def get_out_layers_test(DNNModel,data,hidden_nodes,num_output):
    
    weight, bias = load_weight_bias(DNNModel)
    # data input to HD1
    num_HD_layers = len(hidden_nodes)
    observations = data
    layer_outputs = []

    for i in range(0,num_HD_layers):
        layer_input = np.matmul(observations,weight[i]) + bias[i]
        observations = np.maximum(layer_input, 0, layer_input)
        layer_outputs.append(observations)
        
    
    # output layer
    output = np.matmul(observations,weight[-1]) + bias[-1]
    #layer_outputs.append(output)
    if num_output == 1:
        output = expit(np.resize(output,[1,num_output])) # sigmoid activation
        output = np.round(output)
    else:
        output = softmax(np.resize(output,[1,num_output]))
        output = np.argmax(output)
        
    layer_outputs.append(output)
    
    return(layer_outputs)

''' Main execution '''
if __name__ == '__main__':

    # Get the list of training data filenames
    listdir_test = os.listdir("./NumericalData/ecoli/Data/Testing/")

    # Get the list of DNN model filenames
    listdir_DNNmodel = os.listdir("./NumericalData/ecoli/Model/")

    num_output = 8
    hidden_node_1 = 5
    hidden_node_2 = 5
    hidden_nodes = [hidden_node_1,hidden_node_2]
    num_hd_nodes = sum(hidden_nodes)
    # start ID for choosing DNN model
    startModelID = 0

    # Main loop to execute data operation and get the output at each layer of the DNN
    for i in range(0,len(listdir_test)):
        # Load data
        test_data = np.load("./NumericalData/ecoli/Data/Testing/" + listdir_test[i])
        num_samples = test_data.shape[0]

        # Load DNN model
        for j in range(startModelID, startModelID + 10):
            
            # initialize a writing file and array
            filename = "./NumericalData/ecoli/Data/Layers_Output_Test/" + listdir_DNNmodel[j][0:-3] + ".csv"
            # Initialize output array
            data_out = np.zeros([num_samples,num_hd_nodes+1])
            
            DNNModel = load_model("./NumericalData/ecoli/Model/" + listdir_DNNmodel[j])
            
            # main loop for get output given model and data
            for k in range(0,num_samples):
                layer_outputs = get_out_layers_test(DNNModel,test_data[k,0:-1],hidden_nodes,num_output)
                data_out[k,0:hidden_node_1] = layer_outputs[0]
                data_out[k,hidden_node_1:num_hd_nodes] = layer_outputs[1]
                data_out[k,num_hd_nodes] = layer_outputs[2]
                
            with open(filename, 'w') as f:
                pd.DataFrame(data_out).to_csv(f,encoding='utf-8', index=False, header = False)

        startModelID += 10
        


               