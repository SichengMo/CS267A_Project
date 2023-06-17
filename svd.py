import pandas as pd
import numpy as np
import random
import math
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# Specify the file path for MovieLens data
data_file = 'ml-100k/u5.test'

# Read the MovieLens data into a dataframe
data = pd.read_csv(data_file, names=['user_id', 'movie_id', 'rating', 'timestamp'], delimiter='\t')

num_ratings = len(data)
print('Num ratings: ' + str(num_ratings))

# Create a mapping from the original indices to new indices
user_mapping = {old_id: new_id for new_id, old_id in enumerate(data['user_id'].unique())}
movie_mapping = {old_id: new_id for new_id, old_id in enumerate(data['movie_id'].unique())}

# Update the dataframe with new indices
data['user_index'] = data['user_id'].map(user_mapping)
data['movie_index'] = data['movie_id'].map(movie_mapping)

# Get the number of unique users and movies
num_users = len(user_mapping)
num_movies = len(movie_mapping)

print(num_users)
print(num_movies)

# Create an empty matrix of shape num_users by num_movies
ratings_matrix = np.zeros((num_users, num_movies)) + 3.5  # Mean

print(ratings_matrix)
# assert False

N_test = 2000
test_set = set(random.sample(range(20000), N_test))
test_data = []

# Fill the matrix with ratings
rating_sum = 0
for i, row in data.iterrows():
    user_idx = row['user_index']
    movie_idx = row['movie_index']
    rating = row['rating']
    rating_sum += rating
    assert rating != 0
    if i in test_set:
        test_sample = {}
        test_sample['user_index'] = user_idx
        test_sample['movie_index'] = movie_idx
        test_sample['rating'] = rating
        test_data.append(test_sample)
        # ratings_matrix[user_idx, movie_idx] = rating
    else:
        ratings_matrix[user_idx, movie_idx] = rating

omega = np.where(ratings_matrix == 3.5)
# print(omega)
# print(ratings_matrix)

# assert False

print(rating_sum / 20000)

# Print the shape of the resulting matrix
print(f"The matrix shape is {ratings_matrix.shape}")

# # Create a sparse matrix representation of the ratings
# ratings_matrix = csc_matrix((data['rating'], (data['user_index'], data['movie_index'])), shape=(num_users, num_movies))

# Perform Singular Value Decomposition (SVD)

Z = ratings_matrix
r = 50
k = 100
iters = 200
for i in range(iters):
    U, sigma, Vt = svds(Z, k=k)
    # print(sigma[r+1])
    sigma = sigma - sigma[r+1]
    # print(sigma)
    sigma[sigma<0] = 0
    # print(sigma)
    # assert False
    W = np.dot(np.dot(U, np.diag(sigma)), Vt)
    Z[omega] = W[omega]

ratings_approx = Z

# print(U.shape)
# print(sigma.shape)
# print(Vt.shape)
# assert False

# Construct the low-rank approximation of the ratings matrix
# ratings_approx = np.dot(np.dot(U, np.diag(sigma)), Vt)

sq_error = 0
abs_error = 0
right = 0
x = 0
for test_sample in test_data:
# for test_sample in test_data:
    user_idx = test_sample['user_index']
    movie_idx = test_sample['movie_index']
    rating = test_sample['rating']
    sq_error += (ratings_approx[user_idx][movie_idx] - rating)**2
    x += 1
    if x < 50:
        print('Rating')
        print(ratings_approx[user_idx][movie_idx])
        print(rating)
    abs_error += abs(ratings_approx[user_idx][movie_idx] - rating)
    if round(ratings_approx[user_idx][movie_idx]) == rating:
        right += 1

rmse = math.sqrt(sq_error / N_test)
mae = abs_error / N_test
accuracy = right / N_test
print('RMSE: ' + str(rmse))
print('MAE: ' + str(mae))
print('accuracy: ' + str(accuracy))


# Print the mapping from new indices to original user and movie IDs
# print("User mapping:")
# print(user_mapping)
# print("Movie mapping:")
# print(movie_mapping)
