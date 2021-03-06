import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
    messages, the messages received from different channels after the disaster
    categories, the messages corresponding categories
    Output:
    df the data file after merging the messages and categories tables
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df =  messages.merge(categories, how='outer', on=['id'])
    
    return df


def clean_data(df):
    '''
    Input: 
    df the data file after merging the messages and categories tables
    Output:
    df after cleaning the data, splitting categories columns , giving it new names, adjusting the values to be 0s and 1s and removing duplicates
    '''
    categories = df['categories'].str.split(";" , expand=True)
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =row.str.split("-", expand=True).iloc[:, 0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda a: min(int(a[-1]),1))
    # drop the original categories column from `df`
    df.drop('categories', axis='columns', inplace=True)
    df = pd.concat([df,categories],axis=1)
    # check number of duplicates
    df_duplicate=df[df.duplicated()]
    df_duplicate.shape[0]
    # drop duplicates
    df=df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    '''
    Input: 
    df: the data file the needs to be saved
    database_filename: the database name
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists='replace') 


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
