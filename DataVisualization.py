#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:05:07 2019

Code to answer Part 4: Basic Visualizations

@author: evascheller
"""
import numpy as np
import matplotlib.pyplot as plt

#Problem 1
data = np.loadtxt('data.txt',delimiter='\t')
MovieNames = np.genfromtxt('movies.txt',delimiter='\t',dtype=None,encoding='ISO-8859-1',usecols=(1,))

User_ID = data[:,0]
Movie_ID = data[:,1]
Rating = data[:,2]

labels, counts = np.unique(Rating, return_counts=True)

plt.figure(1)
plt.bar(labels, counts, align='center')
plt.xlabel('Number of Stars')
plt.ylabel('Frequency')
plt.title('All Ratings')
plt.savefig('All_Ratings')

#Problem 2
Movie_ID_number, counts_Movie_ID = np.unique(Movie_ID, return_counts=True)
counts_Movie_ID, Movie_ID_number = zip(*sorted(zip(counts_Movie_ID,Movie_ID_number)))
Most_popular_movie_ID = Movie_ID_number[len(Movie_ID_number)-11:len(Movie_ID_number)-1]
Most_popular_movie_counts = counts_Movie_ID[len(counts_Movie_ID)-11:len(counts_Movie_ID)-1]
Most_popular_movie_name = []
for ID in Most_popular_movie_ID:
    Most_popular_movie_name.append(MovieNames[int(ID)-1])

fig,ax=plt.subplots()
plt.bar([1,2,3,4,5,6,7,8,9,10], Most_popular_movie_counts, align='center')
plt.xlabel('Movie title')
plt.ylabel('Number of Ratings')
plt.title('10 most popular movies')
ax.set_xticks(np.arange(1,11))
ax.set_xticklabels(Most_popular_movie_name,fontsize=4)
plt.savefig('Most_popular_movie')


#Problem 3
Average_rating = []
for ID in Movie_ID_number:
    summing = 0
    division = 0
    for i in range(len(Movie_ID)):
        if Movie_ID[i] == ID:
            summing += Rating[i]
            division += 1
    Average_rating.append(summing/division)
        
Average_rating, Movie_ID_number = zip(*sorted(zip(Average_rating,Movie_ID_number)))

Best_movie_ID = Movie_ID_number[len(Movie_ID_number)-11:len(Movie_ID_number)-1]
Best_movie_rating = Average_rating[len(Average_rating)-11:len(Average_rating)-1]
Best_movie_name = []
for ID in Best_movie_ID:
    Best_movie_name.append(MovieNames[int(ID)-1])
    

fig,ax=plt.subplots()
plt.bar([1,2,3,4,5,6,7,8,9,10], Best_movie_rating, align='center')
plt.xlabel('Movie title')
plt.ylabel('Average Rating')
plt.title('10 best rated movies')
ax.set_xticks(np.arange(1,11))
ax.set_xticklabels(Best_movie_name,fontsize=4)
plt.savefig('Best_movie')

plt.show()


