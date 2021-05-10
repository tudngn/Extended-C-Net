from buildTree_CNet import buildTree_CNet
import numpy as np
import os
import pandas as pd


hidden_node_1 = 5
hidden_node_2 = 5
hidden_nodes = [hidden_node_1,hidden_node_2]
num_hd_nodes = sum(hidden_nodes)

# Find all data files in the working folder
train_data_Directory = "./balance/Data/Layers_Output/"
train_data_File = os.listdir(train_data_Directory)
test_data_Directory = "./balance/Data/Layers_Output_Test/"
test_listFile = os.listdir(test_data_Directory)
test_label_Directory = "./balance/Data/Testing/"
test_labelFile = os.listdir(test_label_Directory)
saveDirectory = "./balance/C5/"

num_training_files = len(train_data_File)

# Learn one tree from each file
filename = saveDirectory + "CNetPerformance_DNN.csv"

for i in range(num_training_files):
    # Load training data
    q, r = divmod(i,10)
    if r == 0:
        test_label_name = test_labelFile[q-1]
    else:
        test_label_name = test_labelFile[q]


    train_filename = train_data_File[i]
    data = np.genfromtxt(fname=train_data_Directory+train_filename, delimiter=",", skip_header=1) 
    trainingData = data[:,hidden_node_1:num_hd_nodes+1]

    # Load testing data
    test_label = np.load(test_label_Directory+test_label_name)
    test_filename = test_listFile[i]
    data = np.genfromtxt(fname=test_data_Directory+test_filename, delimiter=",", skip_header=1)
    testingData = np.hstack((data[:,hidden_node_1:num_hd_nodes], test_label[:,-1]))

    # Build decision trees 
    accuracy, pruned_accuracy, tree_size, pruned_tree_size = buildTree_CNet(train_filename,
                                                                            trainingData,
                                                                            testingData)
    # Append data to write
    dataToWrite = np.array([accuracy,pruned_accuracy,tree_size,pruned_tree_size])

    # Write results to csv file

    if (i == 0):
        header = ["accuracy","pruned_accuracy","tree_size","pruned_tree_size"]
        with open(filename, 'w') as f:
            pd.DataFrame(columns = header).to_csv(f,encoding='utf-8', index=False, header = True)
    
    else:
        with open(filename, 'w') as f:
            pd.DataFrame(dataToWrite).to_csv(f,encoding='utf-8', index=False, header = False)

    