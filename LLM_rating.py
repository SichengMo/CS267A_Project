import json

import pandas as pd
import numpy as np

# Load the files
u5_base = pd.read_csv('ml-100k/u5.base', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
u5_test = pd.read_csv('ml-100k/u5.test', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
u_item = pd.read_csv('ml-100k/u.item', sep='|', names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                             'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                             'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                             'Thriller', 'War', 'Western'], encoding='latin-1')

# Map movie IDs to movie names
id_to_title = u_item.set_index('movie_id')['movie_title'].to_dict()
u5_base['movie_name'] = u5_base['item_id'].map(id_to_title)
u5_test['movie_name'] = u5_test['item_id'].map(id_to_title)

# Create function to get random sample for each user
def sample_movies_for_user(user_id):
    user_data = u5_base[u5_base['user_id'] == user_id]
    sample = user_data.sample(5) if len(user_data) > 5 else user_data
    return sample[['movie_name', 'rating']]

def process_sample(sample):

    content = ''
    for idx, row in sample.iterrows():
        movie_name = row['movie_name']
        rating = row['rating']
        content  += f"Movie Name: {movie_name}, Rating: {rating};"
    return content



content = {}
# For each item in u5.test
for i, row in u5_test.iterrows():
    user_id = row['user_id']
    movie_id = row['item_id']
    score = row['rating']
    movie_name = id_to_title[movie_id]
    print(f"User ID: {user_id}, Movie ID: {movie_id}, Movie Name: {movie_name}")
    print(f"Score:{score}")
    # exit()
    sample = sample_movies_for_user(user_id)
    processed_sample = process_sample(sample)
    query = f"{movie_name}. Reference: {processed_sample}"
    data= {'query': query, 'rating':score}
    content[i]  = data
with open('LLM_eval.json','w') as f:
    json.dump(content,f)



    # print(sample)
