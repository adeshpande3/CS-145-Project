import numpy as np
import pandas as pd
import csv
import code

# Contains info about each business
businessPd = pd.read_csv("data/business.csv")
# Contains two columns, one with the test id and the other with the predictions
# sampleSub = pd.read_csv('sample_submission.csv')
# Contains two columns, one with the business id and the other with the user id
testPairs = pd.read_csv('data/test_queries.csv')
# Contains info about past reviews
# trainReviews = pd.read_csv('train_reviews.csv')
# Contains info about the ratings that users have given in the past
# users = pd.read_csv('users.csv')
# IDK tbh
# validate = pd.read_csv('validate_queries.csv')

with open('firstTry.csv', 'w', newline='') as output:
    testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    testWriter.writerow(['index'] + ['stars'])
    for index,row in enumerate(testPairs.itertuples()):
        biz_id = row.business_id

        # Get stars from businessPd
        biz_stars = businessPd.loc[businessPd['business_id'] == biz_id]['stars'].values[0]
        testWriter.writerow([index] + [biz_stars])
        # code.interact(local=locals())
