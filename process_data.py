import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sqlalchemy import create_engine

y_vars = ['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
'''
INPUT messages_filepath, categories_filepath

RETURN dataframe
'''
    return df


def clean_data(df):
    '''
    INPUT: dataframe
    prepares data for machine learning model
    RETURN: clean dataframe
    '''
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    col_names = categories.query('index == 0').values
    col_names = col_names.tolist()[0]
    # Remove the second to last character onwards of each string with slicing
    for i, element in enumerate(col_names):
        col_names[i] = element[:-2]
    categories.columns = col_names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str.get(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate dataframes
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MainDataFrame', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()