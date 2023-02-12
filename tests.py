import os
import data.process_data as process_data
import pandas as pd


class TestClass:
    """
    Test class for disaster response pipeline.
    Tests the basic scripts of this project
    """

    def test_process_data(self):
        """
        Runs the script process_data.py with the given arguments.
        """
        os.system(('process_data.py disaster_messages.csv '
                   'disaster_categories.csv DisasterResponse.db'))

    def test_load_data(self):
        """
        Tests the load_data function
        """
        process_data.load_data(
            'data\\disaster_messages.csv',
            'data\\disaster_categories.csv'
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

    def test_clean_data(self):
        """
        Test the clean_data function
        """
        # test input
        dict_test = {
            'id': [1, 2, 3, 3],
            '1': 4*[None],
            '2': 4*[None],
            '3': 4*[None],
            'related': [1, 1, 1, 1],
            'request': [1, 2, 0, 0],
            'offer': [0, 1, 1, 1]
        }
        df_test = pd.DataFrame(dict_test)

        # validation dataframe
        df_true = df_test.iloc[[0, 2], :]

        df_cleaned = process_data.clean_data(df_test)

        pd.testing.assert_frame_equal(df_cleaned, df_true, check_dtype=False)

        assert True
