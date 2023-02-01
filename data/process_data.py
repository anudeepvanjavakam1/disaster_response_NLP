import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """This function reads in csv files from messages_filepath and categories_filepath,
       merges both the files on id column and returns a data frame

    Returns:
        df (pd.DataFrame): merged data frame
    """
    # load categories dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')

    return df


def clean_data(df):
    """This function cleans the data frame by creating separate columns (with 0s and 1s) for message categories
       drops duplicates

    Args:
        df (pd.DataFrame): a data frame with categories all in one column

    Returns:
        df (pd.DataFrame): a clean data frame with one column for each category and no duplicates
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.rsplit('-', 1, expand=True)[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis =1)

    #drop categories column
    df = df.drop('categories',axis=1)

    # drop duplicates
    df = df[~df.duplicated()]

    return df

def save_data(df, database_filename):
    """This function saves the data frame as table (messages) in a sqlite db

    Args:
        df (pd.DataFrame)      : a data frame with messages and categories
        database_filename (str): a name for database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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