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
from sklearn import tree
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
# Create Datasets

print "Creating the user dictionary"
userDict = createUserDict()
print "Creating the business dictionary"
businessDict = createBusinessDict()
print "Creating the training set"
X, Y = createTrainingSet()
xTrain, xTest, yTrain, yTest = train_test_split(X, Y)
kaggleTest = createTestSet()

########################################################################################
# Create Models

models = []
#models.append(tree.DecisionTreeClassifier())
#models.append(svm.SVC())
#models.append(KNeighborsClassifier())
#models.append(RandomForestClassifier())
#models.append(tree.DecisionTreeClassifier())
models.append(linear_model.LogisticRegression())

for model in models:
	model.fit(xTrain, yTrain)
	testPreds = model.predict(xTest)
	print "Test accuracy of the model is: {0}".format(np.mean(testPreds == yTest))

########################################################################################
# Get Predictions for Kaggle Test Set

def createKaggleCSV():
	kagglePreds = model.predict(kaggleTest)

	df1 = pd.DataFrame({'labels': kagglePreds})
	kagglePreds = df1['labels'].tolist()
	results = [[0 for x in range(2)] for x in range(len(kagglePreds))]
	for index in range(0,len(kagglePreds)):
	    results[index][0] = int(index)
	    results[index][1] = kagglePreds[index]

	firstRow = [[0 for x in range(2)] for x in range(1)]
	firstRow[0][0] = 'index'
	firstRow[0][1] = 'stars'
	with open("result.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(firstRow)
	    writer.writerows(results)

#createKaggleCSV()