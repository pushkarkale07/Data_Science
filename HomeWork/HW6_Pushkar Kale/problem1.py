import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # generate mat(n_test, n_train)
    D = np.zeros((Xtest.shape[0], Xtrain.shape[0]))
    # go through i, j
    for j_train in range(Xtrain.shape[0]):
        for i_test in range(Xtest.shape[0]):
            # difference sum
            sum = 0
            for p in range(Xtrain.shape[1]):
                sum += np.square(Xtrain[j_train, p] - Xtest[i_test, p])
            D[i_test, j_train] = np.sqrt(sum)

    #########################################
    return D 



#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # Calculate distance
    D = compute_distance(Xtrain, Xtest)

    # Generate the Ytest list
    Ytest = []

    # Go through all the test data
    for i_test in range(Xtest.shape[0]):
        # generate a list of the class value, e.g. class=0, 1, 3, Ytrain_grouplist = [0, 0, 0, 0]
        Ytrain_grouplist = np.zeros((np.max(Ytrain) + 1))
        # sort the distance
        small_list = np.argsort(D[i_test])

        # go through K
        for k in range(K):
            # plus 1 on position of the class number of the k-th smallest distance in the Ytrain_grouplist
            # e.g. k = 1, small_list[k] = 2(the position who has the smallest distance), Ytrain[2] = 3(the class number)
            # original Ytrain_grouplist => [0, 0, 0, 1]
            # Ytrain_grouplist[3] += 1 => [0, 0, 0, 2]
            Ytrain_grouplist[Ytrain[small_list[k]]] = Ytrain_grouplist[Ytrain[small_list[k]]] + 1

        # get the position of the largest number in Ytrain_grouplist, the position index is the class number
        position = np.argmax(Ytrain_grouplist)
        Ytest.append(position)

    # force Ytest transfer to array
    Ytest = np.asarray(Ytest)

    #########################################
    return Ytest 

