import pandas as pd
import numpy as np
import csv
from xgboost import XGBRegressor
from sklearn import linear_model

def createKaggleCSV_vectorML(xTrain, yTrain, kaggleTest, model):
	print ("Creating Kaggle predictions!")
	model.fit(xTrain, yTrain)
	modelPreds = model.predict(kaggleTest)
	kagglePreds = modelPreds
	# If you want to average the preds
	#averageRatingPreds = getAverageRating(testPairs)
	#kagglePreds = [sum(x) / 2 for x in zip(modelPreds, averageRatingPreds)]

	# Threshold the predictions if they go above/below the rating boundaries
	kagglePreds = np.clip(kagglePreds, 1, 5)

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