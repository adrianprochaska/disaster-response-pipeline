import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories data and returns a merged dataframe.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # transform category strings into columns
    categories = str_to_categories(categories)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def str_to_categories(categories):
    """
    Function transforms one column of strings into multiple columns with
    correct category names
    """
    # pop index column
    idx = categories.pop('id')

    # split categories on semicolon
    categories = categories.loc[:, 'categories'].str.split(';', expand=True)

    # rename the columns of `categories`
    categories.columns = get_category_names(categories)

    # remove category name and transform string number into int
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # merge index and new categories dataframe
    categories = pd.concat([idx, categories], axis=1, join='inner')

    return categories


def get_category_names(categories):
    """
    output
    """
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-', expand=True).iloc[:, 0].to_list()
    return category_colnames


def clean_data(df):
    """
    This function cleans the data.
    First, it removes rows with invalid data.
    Second, it drops duplicates.
    """
    # first index of the columns containing categories
    idx_cat_start = 4
    # find rows with invalid data in the category columns
    # values not equal to zero or one are invalid
    idx_cell_invalid = ((df.iloc[:, idx_cat_start:] != 1) &
                        (df.iloc[:, idx_cat_start:] != 0))
    idx_row_invalid = idx_cell_invalid.any(axis=1)

    # drop invalid rows
    df = df.loc[~idx_row_invalid, :]

    # drop duplicates
    df = df.drop_duplicates()

    # remove one-class-columns
    # index of multiclass columns
    idx_multiclass = np.any(
        df.iloc[:, idx_cat_start:] - df.iloc[0, idx_cat_start:] != 0,
        axis=0
    )

    # only change dataframe if a column must be removed
    if any(~idx_multiclass):
        # dataframe with only multiclass columns
        df_multiclass = df.iloc[:, idx_cat_start:].loc[:, idx_multiclass]

        # merge left part of dataframe with multiclass dataframe
        df_return = pd.concat(
            [df.iloc[:, :idx_cat_start], df_multiclass],
            axis=1)
    else:
        df_return = df

    return df_return


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = \
            sys.argv[1:]

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
