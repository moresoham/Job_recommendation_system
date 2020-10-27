# import Flask class from the flask module
from flask import Flask, request
import pickle

# Libraries
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from flask import jsonify

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Enabling CORS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class LemmaTokenizer(object):
    def __init__(self):
        # lemmatize text - convert to base form
        self.wnl = WordNetLemmatizer()
        # creating stopwords list, to ignore lemmatizing stopwords
        self.stopwords = stopwords.words('english')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]


df = pd.read_csv('Dataset/processed_jobs_dataset.csv')
df['RequiredQual'] = df['RequiredQual'].apply(lambda x: x.replace('\n', ' ').replace('\r', '').replace('- ', '').replace(' - ', ' to '))
y = df['Title']
X = df['RequiredQual']

print("X Shape : " + str(X.shape))
print("y Shape : " + str(y.shape))


vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
vectorizer.fit(X)
tfidf_matrix = vectorizer.transform(X)
X_tdif = tfidf_matrix.toarray()
print("tfidf_matrix Shape : " + str(tfidf_matrix.shape))
print("X_tdif Shape : " + str(X_tdif.shape))

enc = LabelEncoder()
enc.fit(y.values)
y_enc = enc.transform(y.values)

X_train_words, X_test_words, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=1)
X_train = vectorizer.transform(X_train_words)
X_train = X_train.toarray()

X_test = vectorizer.transform(X_test_words)
X_test = X_test.toarray()
print(X_test.shape)

def predict(model, input):
    preds_data = {'Current Position': [], 'Current Position Requirments': [], 'Alternative 1': [], 'Alternative 2': []}
    y_preds_proba = model.predict_proba(input)

    counter = 0
    for idx, (pred_row, true_job_position) in enumerate(zip(y_preds_proba, y_test)):
        class_preds = np.argsort(pred_row)
        # delete true class
        for i in [-1, -2]:
            if class_preds[i] == true_job_position:
                class_preds = np.delete(class_preds, i)
        # getting other 2 highest job predictions
        top_classes = class_preds[-2:]
        # obtaining class name string from int label
        class_names = enc.inverse_transform(top_classes)
        true_job_position_name = enc.inverse_transform([true_job_position])
        # saving to dict
        preds_data['Current Position'].append(true_job_position_name[0])
        preds_data['Current Position Requirments'].append(X_test_words.iloc[idx])
        preds_data['Alternative 1'].append(class_names[1])
        preds_data['Alternative 2'].append(class_names[0])
    counter += 1
    return preds_data


@app.route('/rf', methods=['GET', 'POST'])
def rf():
    # Get values
    input_query = request.form.get('input_query')
    print("Input Query" + str(input_query))
    input_query_vectorized = vectorizer.transform([input_query])
    print("Vec Input Query" + str(input_query_vectorized))
    input_query_vectorized = input_query_vectorized.toarray()
    print("Vec Input Query Array" + str(input_query_vectorized))
    preds_df = pd.DataFrame.from_dict(predict(randomforest, input_query_vectorized))
    return jsonify({
        "recommendation_one": preds_df.iloc[0]['Alternative 1'],
        "recommendation_two": preds_df.iloc[0]['Alternative 2']
    })


@app.route('/gnb', methods=['GET', 'POST'])
def gnb():
    # Get values
    input_query = request.form.get('input_query')
    print("Input Query" + str(input_query))
    input_query_vectorized = vectorizer.transform([input_query])
    print("Vec Input Query" + str(input_query_vectorized))
    input_query_vectorized = input_query_vectorized.toarray()
    print("Vec Input Query Array" + str(input_query_vectorized))
    preds_df = pd.DataFrame.from_dict(predict(gnb, input_query_vectorized))
    return jsonify({
        "recommendation_one": preds_df.iloc[0]['Alternative 1'],
        "recommendation_two": preds_df.iloc[0]['Alternative 2']
    })


@app.route('/lr', methods=['GET', 'POST'])
def lr():
    # Get values
    input_query = request.form.get('input_query')
    print("Input Query" + str(input_query))
    input_query_vectorized = vectorizer.transform([input_query])
    print("Vec Input Query" + str(input_query_vectorized))
    input_query_vectorized = input_query_vectorized.toarray()
    print("Vec Input Query Array" + str(input_query_vectorized))
    preds_df = pd.DataFrame.from_dict(predict(logistic, input_query_vectorized))
    return jsonify({
        "recommendation_one": preds_df.iloc[0]['Alternative 1'],
        "recommendation_two": preds_df.iloc[0]['Alternative 2']
    })

'''
@app.route('/nn', methods=['GET', 'POST'])
def nn():
    # Get values
    input_query = request.form.get('input_query')
    print("Input Query" + str(input_query))
    input_query_vectorized = vectorizer.transform([input_query])
    print("Vec Input Query" + str(input_query_vectorized))
    input_query_vectorized = input_query_vectorized.toarray()
    print("Vec Input Query Array" + str(input_query_vectorized))

    global graph
    with graph.as_default():
        preds_df = pd.DataFrame.from_dict(predict(nn, input_query_vectorized))
        return jsonify({
            "recommendation_one": preds_df.iloc[0]['Alternative 1'],
            "recommendation_two": preds_df.iloc[0]['Alternative 2']
        })
'''
# Routes for models Ends

def get_model():
    global randomforest
    global gnb
    global logistic
    

    randomforest_file = open('Models/randomforest_new1.pckl', 'rb')
    randomforest = pickle.load(randomforest_file)
    randomforest_file.close()

    gnb_file = open('Models/gnb_new1.pckl', 'rb')
    gnb = pickle.load(gnb_file)
    gnb_file.close()

    logistic_file = open('Models/logistic_new1.pckl', 'rb')
    logistic = pickle.load(logistic_file)
    logistic_file.close()

    
   
if __name__ == "__main__":
    print("**Starting Server...")

    # Call function that loads Model
    get_model()

# Run Server
get_model()
app.run()
