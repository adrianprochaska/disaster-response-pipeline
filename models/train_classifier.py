import sys

# import libraries
import pandas as pd

from sqlalchemy import create_engine, inspect

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    The function loads the data from a given database.
    It expects the table in the database to be called 'Message'.
    """
    # load dataframe from SQL
    df = load_dataframe(database_filepath)
    # assign input and target data
    X = df.loc[:, 'message']
    idx_start_target = df.columns.get_loc('genre') + 1
    Y = df.iloc[:, idx_start_target:]

    # get category names of target data
    category_names = Y.columns

    return X, Y, category_names


def load_dataframe(database_filename):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # get table name and use the first one
    inspector = inspect(engine)
    table_name = inspector.get_table_names()[0]

    # read the whole data set
    df = pd.read_sql_table(table_name, engine)

    return df


def tokenize(text):
    """
    The function tokenizes the a given string.
    The folling steps are performed.
    - normalize the cases and remove punctuation
    - tokenize the text using nltk's word_tokenize
    - lemmatize text and remove stop words
    """
    # initialize stop words and WordLemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words]

    return tokens


def build_model():
    """
    The function builds a fixed model pipeline and returns it.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None
                )),  # to deactivate warning
                ('tfidf', TfidfTransformer())
            ])),

            # ('txt_len', TextLengthExtractor())
        ])),

        ('clf', MultiOutputClassifier(
            Pipeline([
                ('smt', SMOTE()),
                ('LogReg', LogisticRegressionCV(solver='liblinear'))
            ])
        ))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function evaluates model performance and outputs the f1-score for each
    target variable.
    """
    # predict test set
    Y_test_pred = model.predict(X_test)

    # initialize result list
    f1scores = []
    # calculate and print the F1-score for each predicted category
    for idx, column in enumerate(category_names):
        # calculate F1-score
        f1scores.append(f1_score(Y_test.iloc[:, idx], Y_test_pred[:, idx]))

        # print current F1-score
        print('Column: {}'.format(column))
        print('F1-Score: {}\n'.format(f1scores[-1]))

    return f1scores


def save_model(model, model_filepath):
    """
    The function saves the trained model into model_filepath.
    """
    joblib.dump(model, open(model_filepath, 'wb'))

    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2
        )

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
