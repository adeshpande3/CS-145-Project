import numpy as np
import pandas as pd
import csv
import code

def createKaggleCSV_avgRating(businessPd, testPairs):
	print ("Creating Kaggle predictions!")
	with open('result.csv', 'w', newline='') as output:
	    testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    testWriter.writerow(['index'] + ['stars'])
	    for index,row in enumerate(testPairs.itertuples()):
	        biz_id = row.business_id

	        # Get stars from businessPd
	        biz_stars = businessPd.loc[businessPd['business_id'] == biz_id]['stars'].values[0]
	        testWriter.writerow([index] + [biz_stars])