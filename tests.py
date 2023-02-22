import os
import data.process_data as process_data
import models.train_classifier as train_classifier
import pandas as pd

# from models.train_classifier import tokenize

from sklearn.multioutput import MultiOutputClassifier
# from sklearn.pipeline import FeatureUnion
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline

import joblib


class TestClass:
    """
    Test class for disaster response pipeline.
    Tests the basic scripts of this project
    """

    def test_load_data(self):
        """
        Tests the load_data function
        """
        process_data.load_data(
            'data/disaster_messages.csv',
            'data/disaster_categories.csv'
        )

    def test_str_to_categories(self):
        """
        Tests the str_to_categories function
        """
        # test input
        dict_test = {
            'id': [1, 2],
            'categories': [
                'related-1;request-1;offer-0',
                'related-1;request-2;offer-1'
            ]
        }
        df_test = pd.DataFrame(dict_test)

        # validation dataframe
        dict_true = {
            'id': [1, 2],
            'related': [1, 1],
            'request': [1, 2],
            'offer': [0, 1]
        }
        df_true = pd.DataFrame(dict_true)

        categories = process_data.str_to_categories(df_test)

        pd.testing.assert_frame_equal(categories, df_true, check_dtype=False)

        assert True

    def test_get_category_names(self):
        """
        Tests the get_category_names function
        """
        dict_test = {
            '1': ['related-1', 'related-0'],
            '2': ['request-0', 'request-0'],
            '3': ['offer-0', 'offer-1']
        }

        df_test = pd.DataFrame(dict_test)
        category_names = process_data.get_category_names(df_test)
        assert category_names == ['related', 'request', 'offer']

    def test_clean_data_out_of_range(self):
        """
        Test the clean_data function
        """
        # test input
        dict_test = {
            'id': [1, 2, 3],
            '1': 3*[''],
            '2': 3*[''],
            '3': 3*[''],
            'related': [1, 1, 0],
            'request': [1, 2, 0],
            'offer': [0, 1, 1]
        }
        df_test = pd.DataFrame(dict_test)

        # validation dataframe
        # values should be 1 or 0
        # messages with values out of range shall be removed
        df_true = df_test.iloc[[0, 2], :]

        df_cleaned = process_data.clean_data(df_test)

        pd.testing.assert_frame_equal(df_cleaned, df_true, check_dtype=False)

        assert True

    def test_clean_data_duplicate_rows(self):
        """
        Test the clean_data function
        """
        # test input
        dict_test = {
            'id': [1, 3, 3],
            '1': 3*[''],
            '2': 3*[''],
            '3': 3*[''],
            'related': [1, 0, 0],
            'request': [1, 0, 0],
            'offer': [0, 1, 1]
        }
        df_test = pd.DataFrame(dict_test)

        # validation dataframe
        # duplicate rows shall be removed
        df_true = df_test.iloc[:2, :]

        df_cleaned = process_data.clean_data(df_test)

        pd.testing.assert_frame_equal(df_cleaned, df_true, check_dtype=False)

        assert True

    def test_clean_data_oneclass_column(self):
        """
        Test the clean_data function
        """
        # test input
        dict_test = {
            'id': [1, 2, 3],
            '1': 3*[''],
            '2': 3*[''],
            '3': 3*[''],
            'related': [1, 1, 1],
            'request': [0, 0, 0],
            'offer': [0, 0, 1]
        }
        df_test = pd.DataFrame(dict_test)

        # validation dataframe
        # categories only containing 1s or 0s shall be removed.
        df_true = df_test.drop(['related', 'request'], axis=1)

        df_cleaned = process_data.clean_data(df_test)

        pd.testing.assert_frame_equal(df_cleaned, df_true, check_dtype=False)

        assert True

    def test_load_data_train(self):
        """
        This function tests the load data functionality of the
        train classifier script.
        """
        _, _, category_names = train_classifier.load_data(
            './test_data/DisasterResponse.db'
        )

        assert len(category_names) == 35
        assert category_names[0] == 'related'

    def test_tokenize(self):
        """
        This function tests the tokenize function.
        """
        text_test = 'Can you please help me? Yes, I can!'
        token_out = train_classifier.tokenize(text_test)

        token_validate = ['please', 'help', 'yes']
        assert token_out == token_validate

    def test_build_model(self):
        """
        This function tests the build_model function.
        """

        pipeline = train_classifier.build_model()

        if not isinstance(
            pipeline.named_steps['clf'].estimator.named_steps['smt'],
            SMOTE
        ):
            assert False

    def test_evaluate_model(self):
        """
        This function tests the evaluate_model function.
        """
        # create train data set
        X_train = pd.DataFrame([
            [0],
            [1]
        ])
        y_train = pd.DataFrame([
            [0, 1],
            [1, 0]
        ])

        # create an easy small model
        mdl = MultiOutputClassifier(LogisticRegression()).fit(X_train, y_train)

        # create test data set
        X_test = pd.DataFrame([
            [0],
            [0],
            [1],
            [1]
        ])
        y_pred = pd.DataFrame([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0]
        ])
        y_true = pd.DataFrame([
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1]
        ])

        # calculate F1-score manually
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1score_validate = f1_score(y_true, y_pred, average=None)

        # embed F1-score in print string

        # calculate F1-score using the new funktion
        f1score_test = train_classifier.evaluate_model(
            mdl,
            X_test,
            y_true,
            category_names=['cat1', 'cat2']
        )

        # compare strings
        assert all(f1score_validate == f1score_test)

    def test_save_model(self):
        """
        This function tests the save_model function.
        """
        # load test_model
        test_model_path = './test_data/test_model.pkl'
        model = joblib.load(open(test_model_path, 'rb'))

        # save test_model to a test_path
        save_path = './test_data/pytest_model.pkl'
        train_classifier.save_model(model, save_path)

        # load it again
        model_reload = joblib.load(open(test_model_path, 'rb'))

        # compare models and assert False if they are not the same
        assert isinstance(model_reload, type(model))

        assert model.named_steps.keys() == model_reload.named_steps.keys()

        # remove test_model from test_path
        os.remove(save_path)
