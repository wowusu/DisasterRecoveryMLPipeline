import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
# import graphviz
from IPython.display import Image
# from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
y_vars = ['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

def load_data(database_filepath):
    '''
    INPUT: database_filepath
    
    RETURNS:
    X = disaster messages
    y = categories pf disaster
    df = a pandas dataframe containing the data from the database
    '''
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM MainDataFrame',con = conn)
    X = df['message'].values
    y = df[y_vars].values
    return X, y, df.columns


def tokenize(text, lemmatizer=WordNetLemmatizer(), stop_words = stopwords.words('english')):
     '''
    INPUT: text = disaster message
    
    RETURNS:
    tokens = normalized/ tokenized string
   
    '''
    # remove urls    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = text.strip(url_regex)  
#     normalize
    text = re.sub(r"[a-zA-Z0-9]", " ", text.lower())
#     tokenize        
    tokens = word_tokenize(text)   
#     lemmatize    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
     '''
     This transforms the data by counting tokens and using a TF IDF transformation
    INPUT: n/a
    
    RETURNS:
    pipeline 
   
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])
    return pipeline
    

def evaluate_model(model, X_test, Y_test, category_names):
#     X, y, cols = load_data()
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
     '''
    INPUT: model, X_test, Y_test, category_names
    
    RETURNS:
    model 
   
    '''
    y_pred = model.predict(X_test)
    return model


def save_model(model, model_filepath):
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    
    '''
    saves model as pickle
    '''

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()