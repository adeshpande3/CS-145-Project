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

biz_train_reviews = pd.read_csv("data/biz_sorted_train_reviews.csv")

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

with open('reviews_by_business.csv', 'w', newline='') as output:
    testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    testWriter.writerow(['business_id'] + ['avg_stars_given'] + ['avg_useful'] + ['reviews'])
    
    # Get unique business ids
    bizs = biz_train_reviews.business_id.unique()
    total_num_bizs = len(bizs)

    print("Generating reviews_by_business.csv...")
    for i,bizID in enumerate(bizs):
        # Print progress message
        print("Fetching business {0}/{1}".format(i+1, total_num_bizs), end="\r")

        bizReviews = biz_train_reviews.loc[biz_train_reviews['business_id'] == bizID]
        
        num_bizs = len(bizReviews)
        total_stars_given = 0
        total_useful = 0    
        reviews_unjoined = []    
        
        for biz in bizReviews.itertuples():
            total_stars_given += biz.stars
            total_useful += biz.useful
            reviews_unjoined.append(normalizeText(biz.text))

        avg_stars_given = float(total_stars_given / num_bizs)
        avg_useful = float(total_useful / num_bizs)
        reviews = ','.join(reviews_unjoined)
        reviews_format = "\"" + reviews + "\""

        # Format user id
        bizID_formatted = format_id(bizID)

        row = bizID_formatted + "," + str(avg_stars_given) + "," + str(avg_useful) + "," + reviews_format + "\n"
        
        output.write(row)
