from rpy2.robjects.packages import importr
from rpy2 import robjects
import os
import pickle
import numpy as np
import re

C50 = importr('C50')    
C5_0 = robjects.r('C5.0')
predict = robjects.r('predict')
summary = robjects.r('summary')
#Input the folder of pruned trees 
#OUtput the mean and std of the total number of constraints for one tree
#Output the mean and std of the number of constraints per leaf per tree

num_arguments = 4
tree_Directory = "./balance/C-Net/"
listFile = os.listdir(tree_Directory)
num_constraints_per_tree = []
num_constraints_per_leaf = []

for i in range(len(listFile)):
    if(i%2 != 0):
        with open(tree_Directory+listFile[i], 'rb') as f:
            prunedTree = pickle.load(f)
        
        num_rules = prunedTree.size
        if (num_rules == 0):
            num_constraints = 2
            num_constraints_per_tree.append(num_constraints)
            num_constraints_per_leaf.append(1)    
        else:
            info = summary(prunedTree)
            rule_info = info.output
            # All occurrences of substring in string 
            constraint_location = [i.start() for i in re.finditer("\n\tV", rule_info)]
            num_constraints = len(constraint_location) + num_arguments*num_rules
            num_constraints_per_tree.append(num_constraints)
            num_constraints_per_leaf.append(num_constraints/num_rules)
    

mean_tree_constraints = np.mean(num_constraints_per_tree)
std_tree_constraints = np.std(num_constraints_per_tree)

mean_leaf_constraints = np.mean(num_constraints_per_leaf)
std_leaf_constraints = np.std(num_constraints_per_leaf)


print("leaf constraints = %.2f +- %.2f" %(mean_leaf_constraints, std_leaf_constraints))
print("tree constraints = %.2f +- %.2f" %(mean_tree_constraints, std_tree_constraints))
