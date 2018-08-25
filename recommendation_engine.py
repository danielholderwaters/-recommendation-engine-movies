#recommendation engine 
#written in python
#this begins with 'copied' code, and my unique contribution comes later down after the triple hash
#original code from https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
#dataset from: https://grouplens.org/datasets/movielens/100k/
#cd /cygdrive/c/Users/danie/Desktop/PythonCode -----> this is for Cygwin
#cd /mnt/c/Users/danie/Desktop/PythonCode -----> this is for ConEmu
#cd C:/Users/danie/Desktop/PythonCode ------> this is for windows Command Line
#if you want to run the file in ConEmu

import pandas as pd

#%matplotlib inline
#that last line was hashed because this was originally written in IPython, not Sublimetext.  
#in order to get around this, I use the Python_cmd build system within sublime text

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('c:\\Users\danie\Desktop\PythonCode')
print ("current working directory is:", os.getcwd())

# pass in column names for each CSV as the column name is not given in the file and read them using pandas.
# You can check the column names from the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')
#print(users)
#the above print command is used to test that the code is working properly, nothing more

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#used the following print functions to understand the .iat attribute of a pandas df.  
#later, the method.itertuples calls element 1-3, rather than beginning with 0 and I was confused.  
#the answer to my confusion: in itertuples, the index is element 0.  I'll say that again.  the index is element 0
#print(ratings.head())
#print(ratings.dtypes)
#print(ratings.iat[0,0])
#print(ratings.iat[0,0] ,ratings.iat[0,1], ratings.iat[1,0], ratings.iat[1,1])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')


#print(users.shape)
#print(users.head())
#print(ratings.shape)
#print(ratings.head())
#print(items.shape)
#print(items.head())

#importing the test and train datasets
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
#print(ratings_train.shape, ratings_test.shape)

#BEGIN THE ENGINE!

#finding the number of users and number of movies
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

#Now, we will create a user-item matrix which can be used to calculate the similarity between users and items.
data_matrix = np.zeros((n_users, n_items))


#this next for loop uses the ratings array.  in the user-item matrix called data_matrix, the row corresponds to the user
#and the colum corresponds to the item (movie).  in the ratings array, the user id is column 0 (using python syntax)
#and the movie id is column 1.  the rating is in column 2.  the method itertuples() sees the index as element 0. 
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]


#Now, we will calculate the similarity. We can use the pairwise_distance function from sklearn to calculate the cosine similarity.

from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

#the next two lines are practice in saving the correlation matrices
#np.savetxt("foo.csv", user_similarity, delimiter=",")
#np.savetxt("foo2.csv", item_similarity, delimiter=",")

#This gives us the item-item and user-user similarity in an array form. #The next step is to make predictions based on these similarities.


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#Finally, we will make predictions based on user similarity and item similarity.
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

print(user_prediction.shape)
print(item_prediction.shape)

#I like to save things to make sure they work.  so far, these csv files have no other function.
np.savetxt("foo.csv", user_prediction, delimiter=",")
np.savetxt("foo2.csv", item_prediction, delimiter=",")

#this last line is also just to ensure the whole thing runs
#minor change
print("Program is COMPLETE, mi sybiostro")
