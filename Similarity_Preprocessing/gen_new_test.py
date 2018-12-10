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

def main(input_path, output_path, business_csv_path):
    businesses = pd.read_csv(business_csv_path)
    test_pairs = pd.read_csv(input_path)

    with open(output_path, 'w', newline='') as output:
        testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testWriter.writerow(['business_id'] + ['user_id'] +
                            ['sim_score'] + ['avg_stars_given'] + ['avg_stars_received'])

        total_num_pairs = len(test_pairs.index)

        print("Generating new_test_pairs.csv...")
        for i,test_pair in enumerate(test_pairs.itertuples()):
            # Print progress message
            print("Fetching test pair {0}/{1}".format(i+1, total_num_pairs), end="\r")

            # Format invalid IDs
            userID = format_id(test_pair.user_id)
            bizID = format_id(test_pair.business_id)

            # Collect relevant data
            user = user_reviews.loc[user_reviews['user_id'] == userID]
            biz = business_reviews.loc[business_reviews['business_id'] == bizID]

            try:
                user_review = user.reviews.values[0]
            except:
                user_review =  ""

            try:
                biz_review = biz.reviews.values[0]
            except:
                biz_review = ""

            if user_review == "" or biz_review == "":
                try:
                    avg_stars_received = businesses.loc[businesses['business_id'] == test_pair.business_id].stars.values[0]
                except:
                    print("Could not find business {0}".format(test_pair.business_id))
                    code.interact(local=locals())
                avg_stars_given = avg_stars_received
                score = 0.5

            else:
                avg_stars_given = user.avg_stars_given.values[0]  
                avg_stars_received = biz.avg_stars_given.values[0]  

                # Compute cosine similiarity score
                pairwise = [user_review, biz_review]
                tfidf = TfidfVectorizer().fit_transform(pairwise)
                pairwise_sim = tfidf * tfidf.T
                score = pairwise_sim.data[0]

            # Write new data
            row = bizID + "," + userID + "," + str(score) + "," + str(avg_stars_given) + "," + str(avg_stars_received) + "\n"
            output.write(row)
