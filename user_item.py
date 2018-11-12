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

clean_data = pd.read_csv("data/cleaned_train_reviews.csv", low_memory=False)
users = pd.read_excel('data/users.xlsx')
validation_pairs = pd.read_csv('data/validate_queries.csv')

ratings = [1,2,3,4,5]

def normalizeText(text):
    #TODO Replace slang words

    output_text = text.lower() # Convert to lowercase
    output_text = re.sub(r'[^\w\s]','', output_text) # Remove punctuation
    output_text = re.sub(r'[0-9]+', '', output_text) # Remove numbers
    output_text = output_text.split()
    output_text = [ps.stem(word) for word in output_text if word not in stop_words] # Remove stop words and stem
    output_text = ','.join(output_text)
    return output_text

with open('val_test.csv', 'w', newline='') as output:
    testWriter = csv.writer(output, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # testWriter.writerow(['index'] + ['stars'])
    testWriter.writerow(['user_id'] + ['business_id'] + ['stars'])
    counter = 0
    for index,row in enumerate(validation_pairs.itertuples()):
        # Truncate validation set
        if counter > 50:
            break
        counter += 1

        # Initialize variables
        bizID = row.business_id
        userID = row.user_id
        truth_stars = row.stars
        predicted_stars = 0
        
        # Get user's and business' review histories
        userReviews = clean_data.loc[clean_data['user_id'] == userID]
        bizReviews = clean_data.loc[clean_data['business_id'] == bizID]

        #TODO Returns user's avg ratings if not in training data
        if len(userReviews) == 0: 
            try:
                predicted_stars = users.loc[users['user_id'] == userID]['average_stars'].values[0]
            except:
                code.interact(local=locals())
            testWriter.writerow([userID] + [bizID] + [truth_stars] + [predicted_stars])
            continue
        
        # Normalize each u_rv in userReviews
        # userReviews['text'] = userReviews['text'].apply(lambda x: normalizeText(x))

        # Normalize each b_rv in bizReviews
        # bizReviews['text'] = bizReviews['text'].apply(lambda x: normalizeText(x))

        similarities = np.zeros(len(ratings))
        for i,r in enumerate(ratings):
            rth_similarity = []
            user_r = userReviews.loc[userReviews['stars'] == r]
            biz_r = bizReviews.loc[bizReviews['stars'] == r]
            
            if len(user_r) == 0 or len(biz_r) == 0:
                continue

            # user_reviews = [user_r.text.iloc[j] for j in range(len(user_r))]
            # biz_reviews = [biz_r.text.iloc[j] for j in range(len(biz_r))]
            # u_r_vec = TfidfVectorizer().fit_transform(user_reviews)
            # b_r_vec = TfidfVectorizer().fit_transform(biz_reviews)
            # rth_sim = u_r_vec * b_r_vec.T

            for j in range(len(user_r)):
                for k in range(len(biz_r)):
                    review_pair = [user_r.text.iloc[j], biz_r.text.iloc[k]]
                    tfidf = TfidfVectorizer().fit_transform(review_pair)
                    pairwise_sim = tfidf * tfidf.T
                    rth_similarity.append(pairwise_sim.data[0])

            similarities[i] = max(rth_similarity)
        predicted_stars = np.argmax(similarities) + 1
        testWriter.writerow([userID] + [bizID] + [truth_stars] + [predicted_stars])

        # code.interact(local=locals())
 
        # Get stars from businessPd
        # biz_stars = businessPd.loc[businessPd['business_id'] == biz_id]['stars'].values[0]
        # testWriter.writerow([index] + [biz_stars])
        # code.interact(local=locals())
