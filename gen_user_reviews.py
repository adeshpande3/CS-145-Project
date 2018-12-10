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

user_train_reviews = pd.read_csv("data/user_sorted_train_reviews.csv")

def normalizeText(text):
    #TODO Replace slang words

    output_text = text.lower() # Convert to lowercase
    output_text = re.sub(r'[^\w\s]','', output_text) # Remove punctuation
    output_text = re.sub(r'[0-9]+', '', output_text) # Remove numbers
    output_text = output_text.split()
    output_text = [ps.stem(word) for word in output_text if word not in stop_words] # Remove stop words and stem
    output_text = ','.join(output_text)
    return output_text

def format_id(id):
    if id[0] == '-':
        return id[1:]
    elif id[0] == '=' and id[1] == '-':
        return id[2:]
    elif id[0] == '=' and id[1] != '-':
        return id[1:]
    else:
        return id

with open('reviews_by_user.csv', 'w', newline='') as output:
    testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    testWriter.writerow(['user_id'] + ['avg_stars_given'] + ['avg_useful'] + ['reviews'])
    
    # Get unique user ids
    users = user_train_reviews.user_id.unique()
    total_num_users = len(users)

    print("Generating reviews_by_user.csv...")
    for i,userID in enumerate(users):
        # Print progress message
        print("Fetching user {0}/{1}".format(i+1, total_num_users), end="\r")

        userReviews = user_train_reviews.loc[user_train_reviews['user_id'] == userID]
        
        num_users = len(userReviews)
        total_stars_given = 0
        total_useful = 0    
        reviews_unjoined = []    
        
        for user in userReviews.itertuples():
            total_stars_given += user.stars
            total_useful += user.useful
            reviews_unjoined.append(normalizeText(user.text))

        avg_stars_given = float(total_stars_given / num_users)
        avg_useful = float(total_useful / num_users)
        reviews = ','.join(reviews_unjoined)
        reviews_format = "\"" + reviews + "\""

        # Format user id
        userID_formatted = format_id(userID)

        row = userID_formatted + "," + str(avg_stars_given) + "," + str(avg_useful) + "," + reviews_format + "\n"
        
        output.write(row)

