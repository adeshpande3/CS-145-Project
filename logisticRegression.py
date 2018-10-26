import numpy as np
import pandas as pd
from sklearn import linear_model
import csv

########################################################################################
# Load in CSVs

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

########################################################################################
# Create Training Set

# Create training set 
# Each (x,y) pair
# x = userVector + businessVector
# y = rating

# Creates a vector for each user with the following features
# [avgStars, numReviews]
def createUserDict():
    userDict = {}
    for index, row in users.iterrows():
        avgStars = row['average_stars']
        numReviews = row['review_count']
        userDict[row['user_id']] = [avgStars, numReviews]
    return userDict

# Creates a vector for each business with the following features
# [avgStars, numReviews]
def createBusinessDict():
    businessDict = {}
    for index, row in businessPd.iterrows():
        avgStars = row['stars']
        numReviews = row['review_count']
        businessDict[row['business_id']] = [avgStars, numReviews]
    return businessDict

def createTrainingSet():
	xTrain = []
	yTrain = []
	userDict = createUserDict()
	businessDict = createBusinessDict()
	for index, row in trainReviews.iterrows():
	    uv = userDict[row['user_id']]
	    bv = businessDict[row['business_id']]
	    # Concatenate uv and bv
	    newVector = uv + bv
	    yTrain.append(row['stars'])
	    xTrain.append(newVector)
	return xTrain, yTrain

########################################################################################
# Create Test Set

def createTestSet():
	xTest = []
	for index, row in testPairs.iterrows():
	    uv = userDict[row['user_id']]
	    bv = businessDict[row['business_id']]
	    # Concatenate uv and bv
	    newVector = uv + bv
	    xTest.append(newVector)
	return xTest

########################################################################################
# Create Model

logistic = linear_model.LogisticRegression()
logistic.fit(xTrain, yTrain)
preds = logistic.predict(xTest)

########################################################################################
# Get Predictions for Test Set

df1 = pd.DataFrame({'labels': preds})
preds = df1['labels'].tolist()
results = [[0 for x in range(2)] for x in range(len(preds))]
for index in range(0,len(preds)):
    results[index][0] = int(index)
    results[index][1] = preds[index]

firstRow = [[0 for x in range(2)] for x in range(1)]
firstRow[0][0] = 'index'
firstRow[0][1] = 'stars'
with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(firstRow)
    writer.writerows(results)