# CS267A Fianl Project - Probabilistic Methods for Movie Recommendations

This research paper presents a comprehensive exploration and comparative analysis of a movie recommender system, focusing on the utilization of probabilistic and relational techniques. The domain of movies, characterized by intricate relational structures encompassing cast members, filmmakers, genre classifications, and other hierarchically organized attributes, serves as the basis for enhancing the system's performance. Leveraging the inherent uncertainties associated with evaluating individual user preferences, latent models and Bayesian representations are employed to facilitate effective recommendation generation. In particular, probabilistic matrix factorization and data imputation using low rank approximation were adopted and evaluated for efficacy in the recommender system. All methods were trained and tuned using the MovieLens-100K dataset to ensure robustness and generalizability. Moreover, the project incorporates baseline models such as Deep Neural Networks (DNN) and prompting Large Language Models (LLM). To comprehensively evaluate movie recommender models, we introduce the Multi-Attribute MovieLens dataset (M-MovieLens) which incorporates a wide range of movie-related information including duration, year, directorship, cast members, language, country of origin, and production company affiliations. To automate the annotation process efficiently, a large language model (LLM), ChatGPT, is employed, leveraging its extensive pre-training on a vast array of movie-related textual sources. Evaluation metrics, including Root Mean Square Error (RMSE), accuracy, recall, and precision, are employed to gauge the recommendation system's performance. The study aims to contribute valuable insights into the comparative strengths and weaknesses of relational and probabilistic approaches within the context of movie data analysis.

## Overview

ml-100k is the root folder for MoiveLens-100K dataset.

## Usage
To get our result with DNN, run
```
python dnn.py
```
To get our result with LLM, run
```
python LLM_test.py
```
To get our result with LRA, run
```
python svd.py
```
To get our result with PMF, run PMF.ipynb file.


