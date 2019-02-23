# Basic visualization section, part 4. 

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def makeHistogram(x, numBins, xLabel, yLabel, genTitle):
    plt.hist(x, bins=[1,2,3,4,5,6], ec='black')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(genTitle)
    plt.show()

if __name__ == '__main__':
    # get scifi IDs
    scifi_ID = np.genfromtxt('./data/movies.txt',delimiter='\t',usecols=(0, 17), dtype=None, encoding="ISO-8859-1")
    scifi_ID = scifi_ID[np.where((scifi_ID[:,1] == 1))]

    # get scifi ratings
    ratings = np.loadtxt('./data/data.txt',delimiter='\t')
    print(scifi_ID[0][0])
    print(ratings[0][1])
    scifi_rating_ID = np.isin(ratings[:, 1], scifi_ID[:, 0])
    scifi_ratings = ratings[scifi_rating_ID]
    scifi_ratings = scifi_ratings[:, 2]
    print(scifi_ratings)

    # show scifi ratings
    makeHistogram(scifi_ratings, 5, 'Ratings', 'Frequency', 'Frequency of Scifi Ratings')


    # romance_ID = np.loadtxt('./data/movies.txt',delimiter='\t',usecols = (0, 16))
    # romance_ID = romance_id[np.where((romance_ID[1] == 1))]
    # animation_ID = np.loadtxt('./data/movies.txt',delimiter='\t',usecols = (0, 5))
    # animation_ID = animation_id[np.where((animation_ID[1] == 1))]

    # User_ID = ratings[:,0]
    # Movie_ID = ratings[:,1]
    # Rating = ratings[:,2]

    # labels, counts = np.unique(Rating, return_counts=True)



