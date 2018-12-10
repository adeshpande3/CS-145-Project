import numpy as np
import pandas as pd
import code

import csv
import sys

users = pd.read_csv('data/users.csv')
businessPd = pd.read_csv("data/business.csv")
trainReviews = pd.read_csv("new_train_reviews.csv")
testPairs = pd.read_csv('new_test_queries.csv')

def format_id(id):
    if id[0] == '-':
        return id[1:]
    elif id[0] == '=' and id[1] == '-':
        return id[2:]
    elif id[0] == '=' and id[1] != '-':
        return id[1:]
    else:
        return id

def createUserDict():
    userDict = {}
    for index, row in users.iterrows():
        avgStars = row['average_stars']
        numReviews = row['review_count']
        fans = row['fans']
        useful = row['useful']
        funny = row['funny']
        cool = row['cool']
        userDict[format_id(row['user_id'])] = [avgStars, numReviews, fans, useful, funny, cool]
    return userDict

# Creates a vector for each business with the following features
# [avgStars, numReviews]
def createBusinessDict():
    businessDict = {}
    for index, row in businessPd.iterrows():
        avgStars = row['stars']
        numReviews = row['review_count']
        businessDict[format_id(row['business_id'])] = [avgStars, numReviews]
    return businessDict

def createTrainingSet():
    xTrain = []
    yTrain = []
    total_num = len(trainReviews)

    for index, row in trainReviews.iterrows():
        print("Generating training pair {0}/{1}".format(index+1, total_num), end="\r")
        
        try:
            uv = userDict[format_id(row['user_id'])]
        except:
            continue
        try:
            bv = businessDict[format_id(row['business_id'])]
        except:
            continue

        # Get sim score
        sim_score = row['sim_score']
        avg_stars_given = row['avg_stars_given']
        avg_stars_received = row['avg_stars_received']

        # Concatenate uv and bv
        newVector = uv + bv + [sim_score] + [avg_stars_given] + [avg_stars_received]

        yTrain.append(row['stars'])
        xTrain.append(newVector)
    return np.asarray(xTrain), np.asarray(yTrain)

def createTestSet():
    xTest = []
    total_num = len(testPairs)

    for index, row in testPairs.iterrows():
        print("Generating test pair {0}/{1}".format(index+1, total_num), end="\r")

        try:
            uv = userDict[row['user_id']]
        except:
            # print("Can't find user {0}".format(row['user_id']))
            # code.interact(local=locals())
            continue
        try:
            bv = businessDict[row['business_id']]
        except:
            # print("Can't find biz {0}".format(row['business_id']))
            # code.interact(local=locals())
            continue
        # Get sim score
        sim_score = row['sim_score']
        avg_stars_given = row['avg_stars_given']
        avg_stars_received = row['avg_stars_received']

        # Concatenate uv and bv
        newVector = uv + bv + [sim_score] + [avg_stars_given] + [avg_stars_received]
        xTest.append(newVector)
    return xTest

print("Creating the user dictionary")
userDict = createUserDict()
print("Creating the business dictionary")
businessDict = createBusinessDict()
print("Creating the training set")
X, Y = createTrainingSet()
print("Creating the testing set")
kaggleTest = createTestSet()


np.save('xTrain', X)
np.save('yTrain', Y)
np.save('kaggleTest', kaggleTest)