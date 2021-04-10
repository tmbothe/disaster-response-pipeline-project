import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """This function load the data from disk and return data frame
       merge and return a data frame 
    Args:
        messages_filepath ([string]): Path to the message file
        categories_filepath ([string]): Path to the category file
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='inner', on='id')

    return df


def clean_data(df):
    """This function clean the data frame and return a cleaner dataframe

    Args:
        df : Data frame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[1, :]

    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(
            lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)

    categories = categories.clip(0, 1)

    one_value_columns = [column for column in categories.columns if len(
        categories[column].unique()) == 1]
    categories = categories.drop(one_value_columns, axis=1)

    df.drop(columns=['categories'], index=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def save_data(df, database_filename):
    """This function saves the dataframe to a  sqllite databae
    Args:
        df : Data frame
        database_filename : Database name
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
