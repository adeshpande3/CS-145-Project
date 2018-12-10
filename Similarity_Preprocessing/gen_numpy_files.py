import numpy as np
import pandas as pd
import code

import csv
import sys

trainReviews = pd.read_csv("new_train_reviews.csv")
new_testPairs = pd.read_csv("new_test_queries.csv")

def format_id(id):
    if id[0] == '-':
        return id[1:]
    if id[0] == '=' and id[1] == '-':
        return id[2:]
    elif id[0] == '=' and id[1] != '-':
        return id[1:]
    else:
        return id

def createUserDict(users):
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
def createBusinessDict(businessPd):
    businessDict = {}
    for index, row in businessPd.iterrows():
        avgStars = row['stars']
        numReviews = row['review_count']
        businessDict[row['business_id']] = [avgStars, numReviews]
    return businessDict

def createTrainingSet(userDict, businessDict):
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

def createTestSet(old_testPairs, old_testMatrix):
    xTest = []
    total_num_new = len(new_testPairs)
    total_num_old = len(old_testPairs)
    assert(total_num_new == total_num_old)

    new_testMatrix = []

    for index, row in enumerate(old_testMatrix):
        print("Generating test pair {0}/{1}".format(index+1, total_num_old), end="\r")

        new_row = new_testPairs.iloc[index]

        sim_score = new_row['sim_score']
        avg_stars_given = new_row['avg_stars_given']
        avg_stars_received = new_row['avg_stars_received']

        extra = [sim_score, avg_stars_given, avg_stars_received]
        newVec = list(row) + extra

        new_testMatrix.append(newVec)

    return np.asarray(new_testMatrix)

def main(users_path, business_path, tests_path, old_test_matrix_path):
    # Load files
    users = pd.read_csv(users_path)
    businessPd = pd.read_csv(business_path)
    old_testPairs = pd.read_csv(tests_path)
    old_testMatrix = np.load(old_test_matrix_path) # Currently does not support generating old test matrix on the spot because of janky IDs, please perform data cleaning

    print("Creating the user dictionary")
    userDict = createUserDict(users)
    print("Creating the business dictionary")
    businessDict = createBusinessDict(businessPd)

    print("Creating the training set")
    X, Y = createTrainingSet(userDict, businessDict)

    print("Creating the testing set")
    kaggleTest = createTestSet(old_testPairs, old_testMatrix)


    np.save('xTrain', X)
    np.save('yTrain', Y)
    np.save('kaggleTest', kaggleTest)

    return X, Y, kaggleTest