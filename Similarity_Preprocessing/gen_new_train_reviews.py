import numpy as np
import pandas as pd
import csv
import code
import re

from stop_words import stop_words # Defined in http://www.lextek.com/manuals/onix/stopwords1.html
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()

user_dtype = {"user_id": object, 
                "avg_stars_given": float, 
                "avg_useful": float, 
                "reviews": object}

biz_dtype = {"business_id": object, 
                "avg_stars_given": float, 
                "avg_useful": float, 
                "reviews": object}


user_reviews = pd.read_csv("reviews_by_user.csv", dtype=user_dtype, encoding="ISO-8859-1")
business_reviews = pd.read_csv("reviews_by_business.csv", dtype=biz_dtype, encoding="ISO-8859-1")

def format_id(id):
    if id[0] == '-':
        return id[1:]
    elif id[0] == '=' and id[1] == '-':
        return id[2:]
    elif id[0] == '=' and id[1] != '-':
        return id[1:]
    else:
        return id

def main(input_path, output_path):
    with open(output_path, 'w', newline='') as output:
        train_reviews = pd.read_csv(input_path)

        testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testWriter.writerow(['business_id'] + ['cool'] + ['funny'] + ['useful'] + ['stars'] + ['user_id'] +
                            ['sim_score'] + ['avg_stars_given'] + ['avg_stars_received'])

        total_num_pairs = len(train_reviews.index)

        print("Generating new_train_reviews.csv...")
        for i,train_data in enumerate(train_reviews.itertuples()):
            # Print progress message
            print("Fetching training pair {0}/{1}".format(i+1, total_num_pairs), end="\r")

            if i < 100554:
                continue

            # Format invalid IDs
            userID = format_id(train_data.user_id)
            bizID = format_id(train_data.business_id)

            # Collect relevant data
            user = user_reviews.loc[user_reviews['user_id'] == userID]

            try:
                user_review = user.reviews.values[0]
            except:
                print("Can't find userID {0}".format(userID))
                code.interact(local=locals())

            # user_review = user.reviews.values[0]
            avg_stars_given = user.avg_stars_given.values[0]   
            biz = business_reviews.loc[business_reviews['business_id'] == bizID]

            try:
                biz_review = biz.reviews.values[0]
            except:
                print("Can't find bizID {0}".format(bizID))
                code.interact(local=locals())
            
            avg_stars_received = biz.avg_stars_given.values[0]   

            # Collect old data
            cool = train_data.cool
            funny = train_data.funny
            useful = train_data.useful
            stars = train_data.stars

            # Compute cosine similiarity score
            pairwise = [user_review, biz_review]
            tfidf = TfidfVectorizer().fit_transform(pairwise)
            pairwise_sim = tfidf * tfidf.T
            score = pairwise_sim.data[0]

            # Write new data
            row = bizID + "," + str(cool) + "," + str(funny) + "," + str(useful) + "," + str(stars) + "," + userID + "," + str(score) + "," + str(avg_stars_given) + "," + str(avg_stars_received) + "\n"
            output.write(row)



        
