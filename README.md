# Extended-C-Net
Extended C-Net Algorithm

The C-Net algorithm,  proposed  by  Abbass  et  al.  (2001),  is  one  of  those  early  algorithmswhich uses a univariate decision tree (UDT) to generate a multivariate decisiontree (MDT) from neural networks. We propose a modification of the algorithm into an extended version of C-Net to extract the rules from deeper ANNs. 

The multivariate decision tree Extended C-Net is a rewrite of the ANN intointerpretable logic using nodes/conditions and branches/information-flows.  Each rule induced by each leaf of the resultant Extended C-Net tree can be traversed back to weights of the network to deduce the input-target relationships as represented by the network and its weights. 

The procedures of Extended C-Netalgorithm can be described as follows:
- Step 1:Feed inputs to ANN and compute the values at final hidden layer’s nodes
- Step 2:Use values above and labels of instances to train a UDT (C5 decision trees) and extract rules from UDT tree.
- Step 3:Build a decision tree by traverse back to weights of the network in previous layers to deduce the input-target relationships.
- Step 4:Extract final set of rules for every leaf.

The full algorithm can be found in the paper: "Towards Interpretable Neural Networks: An Exact Transformation to Multi-Class Multivariate Decision Trees"

Reference:

[1] Abbass, H. A., Towsey, M., & Finn, G. (2001).  C-Net:  A method for generat-ing non-deterministic and dynamic multivariate decision trees.Knowledge andInformation Systems,3, 184–197.

[2] Nguyen, T.D., Kasmarik, K.E. and Abbass, H.A., 2020. Towards interpretable deep neural networks: An exact transformation to multi-class multivariate decision trees. arXiv e-prints, pp.arXiv-2003.


