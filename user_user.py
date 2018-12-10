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

clean_data = pd.read_csv("data/cleaned_train_reviews.csv")
users = pd.read_csv('data/users.csv')
validation_pairs = pd.read_csv('data/validate_queries.csv')
businesses = pd.read_csv("data/business.csv")
ratings = [1,2,3,4,5]

# =SQRT(SUMSQ(E2:E1002)/COUNTA(E2:E1002))

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
    testWriter.writerow(['user_id'] + ['business_id'] + ['stars'] + ['predicted'])
    counter = 1
    for index,row in enumerate(validation_pairs.itertuples()):
        # Truncate validation set
        if counter > 100:
            break
        print("Computing val_pair {0} of 100...".format(counter))
        counter += 1

        # Initialize variables
        bizID = row.business_id
        userID = row.user_id
        truth_stars = row.stars
        predicted_stars = 0
        
        # Get user's and business' review histories
        userReviews = clean_data.loc[clean_data['user_id'] == userID]
        bizReviews = clean_data.loc[clean_data['business_id'] == bizID]
        biz_stars = businesses.loc[businesses['business_id'] == bizID]['stars'].values[0]

        # If no reviews, return avg stars of the business
        if len(bizReviews) == 0 or len(userReviews) == 0:
            try:
                user_stars = users.loc[users['user_id'] == userID]['average_stars'].values[0]
            except:
                user_stars = biz_stars
            predicted_stars = (biz_stars + user_stars)/2
            testWriter.writerow([userID] + [bizID] + [truth_stars] + [predicted_stars])
            continue

        # Compute user-user similarity vector
        w_ui_u = np.zeros(len(bizReviews))
        ui_avgs = np.zeros(len(bizReviews))
        ui_ratings = np.zeros(len(bizReviews))
        for i,bizReview in enumerate(bizReviews.itertuples()):
            ui = bizReview.user_id
            ui_ratings[i] = bizReview.stars
            try:
                ui_avgs[i] = users.loc[users['user_id'] == ui]['average_stars'].values[0]
            except:
                ui_avgs[i] = bizReview.stars
            uiReviews = clean_data.loc[clean_data['user_id'] == ui]
            similarities = np.zeros(len(ratings))
            for j,r in enumerate(ratings):
                # rth_similarity = []
                userReviews_r = userReviews.loc[userReviews['stars'] == r]
                uiReviews_r = uiReviews.loc[uiReviews['stars'] == r]

                if len(userReviews_r) == 0 or len(uiReviews_r) == 0:
                    continue

                # Compute r-th similarity
                u_text, ui_text = "",""
                u_text = ' '.join([u_text + text for text in userReviews_r.text])
                ui_text = ' '.join([ui_text + text for text in uiReviews_r.text])
                pairwise = [normalizeText(u_text), normalizeText(ui_text)]
                tfidf = TfidfVectorizer().fit_transform(pairwise)
                pairwise_sim = tfidf * tfidf.T
                similarities[j] = pairwise_sim.data[0]
                
            w_ui_u[i] = np.argmax(similarities) + 1

        # Compute prediction
        user_avg = users.loc[users['user_id'] == userID]['average_stars'].values[0]
        sim_sum = np.sum(w_ui_u)
        predicted_stars = user_avg + np.sum(w_ui_u * (ui_ratings - ui_avgs))/sim_sum       
        testWriter.writerow([userID] + [bizID] + [truth_stars] + [predicted_stars])
        
        # Compute error
