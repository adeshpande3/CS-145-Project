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
import Similarity_Preprocessing as sp

from vectorMLModels import createKaggleCSV_vectorML
from avgBusinessRating import createKaggleCSV_avgRating

########################################################################################
# Load in CSVs

# Contains info about each business
businessPd = pd.read_csv("data/business.csv")
# Contains two columns, one with the test id and the other with the predictions
sampleSub = pd.read_csv('data/sample_submission.csv')
# Contains two columns, one with the business id and the other with the user id
testPairs = pd.read_csv('data/test_queries.csv')
# Contains info about past reviews
trainReviews = pd.read_csv('data/train_reviews.csv')
# Contains info about the ratings that users have given in the past
users = pd.read_csv('data/users.csv')

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
# Create/Loading in Datasets

print("\n#############################################################################")
print ("\nHello! If you'd like to run this code, you can either choose to load in \
precomputed matrices for X, Y, and XTest. Or you can choose to create new ones \
using our preprocessing functions. You also have the choice of including or not \
including our similarity score features. \n")

loadIn = raw_input("Do you want to load in precomputed training and testing matrices (y/n)?")
simScore = raw_input("Do you want to include the similarity score features (y/n)?")
modelChoice = raw_input("Which model (1 - AvgRating, 2 - LogReg, 3 - GBRT)?")
if loadIn == "y":
	print("Loading training and testing matrices")
	if simScore == "y":
		X = np.load('Precomputed Matrices/similarity_score_xTrain.npy')
		Y = np.load('Precomputed Matrices/similarity_score_yTrain.npy')
		kaggleTest = np.load('Precomputed Matrices/similarity_score_kaggleTest.npy')
	else:
		X = np.load('Precomputed Matrices/original_xTrain.npy')
		Y = np.load('Precomputed Matrices/original_yTrain.npy')
		kaggleTest = np.load('Precomputed Matrices/original_kaggleTest.npy')
else:
	print("Creating training and testing matrices")
	if simScore == "y":
		# Warning: IDs may not be valid across all data files, please consider cleaning your data first.
		sp.gen_biz_reviews.main("data/business.csv", "reviews_by_business.csv")
		sp.gen_user_reviews.main("data/users.csv", "reviews_by_users.csv")
		sp.gen_new_train_reviews.main("data/train_reviews.csv", "new_train_reviews.csv")
		sp.gen_new_test.main("data/test_queries.csv", "new_test_queries.csv", "data/business.csv")
		X, Y, kaggleTest = sp.gen_numpy_files.main("data/users.csv", "data/business.csv", "data/test_queries.csv", "Precomputed Matrices/previousKaggleTest.npy")
		
	else:	
		print "Creating the user dictionary"
		userDict = createUserDict()
		print "Creating the business dictionary"
		businessDict = createBusinessDict()
		print "Creating the training set"
		X, Y = createTrainingSet()
		print "Creating the testing set"
		kaggleTest = createTestSet()

xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

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

if modelChoice == "2":
	# Logistic Regression!
	# Testing our model on our held out section of the Kaggle training data
	model = linear_model.LogisticRegression()
	model.fit(xTrain, yTrain)
	testPreds = model.predict(xTest)
	print "Validation accuracy of the model is: {0}".format(np.mean(np.round(testPreds) == yTest))
	# Creating our Kaggle predictions
	createKaggleCSV_vectorML(X, Y, kaggleTest, linear_model.LogisticRegression())
elif modelChoice == "1":
	# Average Rating!
	createKaggleCSV_avgRating(businessPd, testPairs)
else:
	# Gradient Boosted Regression Tree!
	# Testing our model on our held out section of the Kaggle training data
	model = XGBRegressor()
	model.fit(xTrain, yTrain)
	testPreds = model.predict(xTest)
	print "Validation accuracy of the model is: {0}".format(np.mean(np.round(testPreds) == yTest))
	# Creating our Kaggle predictions
	createKaggleCSV_vectorML(X, Y, kaggleTest, XGBRegressor())
