import numpy as np
import pandas as pd

# Contains info about each business
businessPd = pd.read_csv("business.csv")
# Contains two columns, one with the test id and the other with the predictions
sampleSub = pd.read_csv('sample_submission.csv')
# Contains two columns, one with the business id and the other with the user id
testPairs = pd.read_csv('test_queries.csv')
# Contains info about past reviews
trainReviews = pd.read_csv('train_reviews.csv')
# Contains info about the ratings that users have given in the past
users = pd.read_csv('users.csv')
# IDK tbh
validate = pd.read_csv('validate_queries.csv')

