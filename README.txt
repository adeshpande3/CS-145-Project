########################################################################################
Group Name: The Least Squares (ID #12)
Members: Adit Deshpande, Elena Escalas, Nina Maller, Allen Miyazawa, Kai Wong

mainScript.py is our top level CS 145 project python script. This script is created so that you
can choose to either load in precomputed training/testing matrices or create them
yourselves through our preprocessing functions. The script also allows you to 
choose whether or not to include our similarity features into the training/testing
matrices. You will also be asked which model you would like to train. Our 3 options
are AvgRating, Logistic Regression, and GBRT. 

Regardless of those choices, the script will test the model on a validation dataset
and create a kaggle prediction file called result.csv. 

We have a couple of different helper scripts that we explain here:

- vectorMLModels.py: Python script that takes in training and testing matrices as
well as an argument for the type of Sklearn model that we want to train on. 
- avgBusinessRating.py: Python script that outputs that average business rating for 
each of the examples in test_queries.
- Functions in Simalirity Preprocessing: Scripts that create training/testing 
matrices that include similarity metrics between business and user. 

This code has been tested and run with Python 2!

Sample Usage:
	python mainScript.py

*Be sure to have all the CSVs from Kaggle in a folder called data.
 - business.csv, sample_submission.csv, test_queries.csv, train_reviews.csv, users.csv
*

########################################################################################