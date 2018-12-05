########################################################################################
'''
Group Name: The Least Squares (ID #12)
Members: Adit Deshpande, Elena Escalas, Nina Maller, Allen Miyazawa, Kai Wong

This is our top level CS 145 project python script. Our project is structured such 
that the 4 algorithms we use are located in different files that we load in. 

- vectorMLModels.py: Python script 
- avgBusinessRating.py: Python script that outputs that average business rating for 
each of the examples in test_queries.
- 

All of these scripts will output a result.csv file that contains the predictions for 
the respective models that are used. 

Sample Usage:
	python ___
'''

########################################################################################
# All Imports
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn import tree
import csv
import sys

from vectorMLModels import createKaggleCSV_vectorML
from avgBusinessRating import createKaggleCSV_avgRating

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
# [avgStars, numReviews, fans, useful, funny, cool]
def createUserDict():
    userDict = {}
    for index, row in users.iterrows():
        avgStars = row['average_stars']
        numReviews = row['review_count']
        fans = row['fans']
        useful = row['useful']
        funny = row['funny']
        cool = row['cool']
        userDict[row['user_id']] = [avgStars, numReviews, fans, useful, funny, cool]
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
	for index, row in trainReviews.iterrows():
	    uv = userDict[row['user_id']]
	    bv = businessDict[row['business_id']]
	    # Concatenate uv and bv
	    newVector = uv + bv
	    yTrain.append(row['stars'])
	    xTrain.append(newVector)
	return np.asarray(xTrain), np.asarray(yTrain)

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
# Helper Functions

# Returns just the average rating of the business in question
def getAverageRating(testPairs):
	avgRating = []
	for index, row in testPairs.iterrows():
		bv = businessDict[row['business_id']]
		avgRating.append(bv[0])
	return np.asarray(avgRating)

########################################################################################
# Create Datasets

print "Creating the user dictionary"
userDict = createUserDict()
print "Creating the business dictionary"
businessDict = createBusinessDict()
print "Creating the training set"
X, Y = createTrainingSet()
xTrain, xTest, yTrain, yTest = train_test_split(X, Y)
print "Creating the testing set"
kaggleTest = createTestSet()

########################################################################################
# Create Models
# This section is for testing out our models on our own test/train split and observing 
# the results. 

models = []
#models.append(tree.DecisionTreeClassifier())
#models.append(svm.SVC())
#models.append(KNeighborsClassifier())
#models.append(RandomForestClassifier())
#models.append(tree.DecisionTreeClassifier())
#models.append(linear_model.LogisticRegression())
#models.append(XGBRegressor())

for model in models:
	model.fit(xTrain, yTrain)
	testPreds = model.predict(xTest)
	print "Test accuracy of the model is: {0}".format(np.mean(np.round(testPreds) == yTest))

########################################################################################
# Get Predictions for Kaggle Test Set

finalModels = ['Similarity', 'Linear Regression', 'Average Rating', 'Gradient Boosted Regression Tree']
# You can change this!!
modelToUse = finalModels[2]

if modelToUse == 'Similarity':
	# TODO
	pass
elif modelToUse == 'Linear Regression':
	createKaggleCSV(X, Y, kaggleTest, linear_model.LogisticRegression())
elif modelToUse == 'Average Rating':
	createKaggleCSV_avgRating(businessPd, testPairs)
else:
	createKaggleCSV(X, Y, kaggleTest, XGBRegressor())
