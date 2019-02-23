#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:05:07 2019

@author: evascheller
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./data/data.txt',delimiter='\t')
#movies = np.loadtxt('./data/movies.txt',delimiter='\t',usecols = (0, 2))

User_ID = data[:,0]
Movie_ID = data[:,1]
Rating = data[:,2]

labels, counts = np.unique(Rating, return_counts=True)

plt.figure(1)
plt.bar(labels, counts, align='center')
plt.xlabel('Number of Stars')
plt.ylabel('Frequency')
plt.title('All Ratings')
plt.show()




