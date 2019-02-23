#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:48:55 2019

@author: evascheller
"""
import numpy as np
from scikit.surprise import SVD

training_data = np.loadtxt('train.txt',delimiter='\t')
test_data = np.loadtxt('test.txt',delimiter='\t')
User_ID_training = training_data[:,0]
Movie_ID_training = training_data[:,1]
Rating_training = training_data[:,2]
User_ID_test = test_data[:,0]
Movie_ID_test = test_data[:,1]
Rating_test = test_data[:,2]






