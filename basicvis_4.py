# Basic visualization section, part 4. 

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# kind of a bar chart, but histogram nonetheless
def makeHistogram(occurences, xLabel, yLabel, genTitle):
    labels, counts = np.unique(occurences, return_counts=True)

    plt.bar(labels, counts)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(genTitle)
    plt.show()

# gets all genre ratings
# takes the moviedata array, rating data array, and the column in 
# movie data array that corresposnds to this genre. 
def getGenreRatings(moviedata_fname, ratingdata_fname, genre_col):
    # get scifi IDs
    ID = np.genfromtxt(moviedata_fname,delimiter='\t',usecols=(0, genre_col), dtype=None, encoding="ISO-8859-1")
    ID = ID[np.where((ID[:,1] == 1))]

    # get scifi ratings
    ratings = np.loadtxt(ratingdata_fname,delimiter='\t')
    rating_ID = np.isin(ratings[:, 1], ID[:, 0])
    genre_ratings = ratings[rating_ID]
    genre_ratings = genre_ratings[:, 2]

    return genre_ratings

if __name__ == '__main__':
    # get and show scifi ratings
    scifi_ratings = getGenreRatings('./data/movies.txt', './data/data.txt', 17)
    makeHistogram(scifi_ratings, 'Ratings', 'Frequency', 'Frequency of Scifi Ratings')

    romance_ratings = getGenreRatings('./data/movies.txt', './data/data.txt', 16)
    makeHistogram(romance_ratings, 'Ratings', 'Frequency', 'Frequency of Romance Ratings')

    animation_ratings = getGenreRatings('./data/movies.txt', './data/data.txt', 5)
    makeHistogram(animation_ratings, 'Ratings', 'Frequency', 'Frequency of Animation Ratings')



