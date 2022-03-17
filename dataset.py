"""dataset.py: """

import random
from os.path import join as oj

import numpy as np
import pandas as pd
from joblib import Memory

import helper


class Dataset:
    """
    All functions take **kwargs, so you can specify any judgement calls you aren't sure about with a kwarg flag.
    Please refrain from shuffling / reordering the data in any of these functions, to ensure a consistent test set.
    """

    def clean_data(self, data_path: str, **kwargs):
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated).
        Data types should be reasonable.

        Params
        ------
        data_path: str, optional
            Path to all data files
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        cleaned_data: dictionary of input and output dictionaries (contain years as dataframes)
        """
        # getting data paths
        io_current_path = oj(data_path, 'raw/AllTablesIO/IOUse_After_Redefinitions_PRO_1997-2020_Summary.xlsx')
        io_old_path = oj(data_path, 'raw/1947-1997-Historical/IOUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx')
        io_new_path = oj(data_path, 'raw/1947-1997-Historical/IOUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx')
        supply_path = oj(data_path, 'raw/AllTablesSUP/Supply_1997-2020_SUM.xlsx')
        make_old_path = oj(data_path, 'raw/1947-1997-Historical/IOMake_Before_Redefinitions_1947-1962_Summary.xlsx')
        make_new_path = oj(data_path, 'raw/1947-1997-Historical/IOMake_Before_Redefinitions_1963-1996_Summary.xlsx')

        # data frames created from tests
        io_current = pd.read_excel(io_current_path, sheet_name=None)
        io_old = pd.read_excel(io_old_path, sheet_name=None)
        io_new = pd.read_excel(io_new_path, sheet_name=None)
        supply = pd.read_excel(supply_path, sheet_name=None)
        make_old = pd.read_excel(make_old_path, sheet_name=None)
        make_new = pd.read_excel(make_new_path, sheet_name=None)

        # run helper function to rename columns
        io_current = helper.rename_columns(io_current, is_io=True, is_historical=False)
        io_old = helper.rename_columns(io_old, is_io=True, is_historical=True)
        io_new = helper.rename_columns(io_new, is_io=True, is_historical=True)
        supply = helper.rename_columns(supply, is_io=False, is_historical=False)
        make_old = helper.rename_columns(make_old, is_io=True, is_historical=True)
        make_new = helper.rename_columns(make_new, is_io=True, is_historical=True)

        # run helper function to replace ... with NaN
        io_current = helper.replace_dots(io_current)
        io_old = helper.replace_dots(io_old)
        io_new = helper.replace_dots(io_new)
        supply = helper.replace_dots(supply)
        make_old = helper.replace_dots(make_old)
        make_new = helper.replace_dots(make_new)

        # take all and put into a dictionary
        cleaned_data = {}
        cleaned_data['io_current'] = io_current
        cleaned_data['io_old'] = io_old
        cleaned_data['io_new'] = io_new
        cleaned_data['supply'] = supply
        cleaned_data['make_old'] = make_old
        cleaned_data['make_new'] = make_new

        return cleaned_data

    def preprocess_data(self, cleaned_data, **kwargs):
        """
        Preprocess the data.
        Impute missing values.
        Scale/transform values.
        Should put the prediction target in a column named "outcome"

        Parameters
        ----------
        cleaned_data: Dict(Dict(pd.DataFrames), Dict(pd.DataFrame))
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        preprocessed_data: Dict(Dict(pd.DataFrames), Dict(pd.DataFrame))
        """

        preprocessed_data = cleaned_data
        preprocessed_data['io_current'] = helper.replace_zeros(cleaned_data['io_current'], True)
        preprocessed_data['io_old'] = helper.replace_zeros(cleaned_data['io_old'], True)
        preprocessed_data['io_new'] = helper.replace_zeros(cleaned_data['io_new'], True)
        preprocessed_data['supply'] = helper.replace_zeros(cleaned_data['supply'], False)
        preprocessed_data['make_old'] = helper.replace_zeros(cleaned_data['make_old'], True)
        preprocessed_data['make_new'] = helper.replace_zeros(cleaned_data['make_new'], True)

        return preprocessed_data

    def extract_features(self, preprocessed_data, **kwargs):
        """
        Extract features from preprocessed data.
        All features should be binary.

        Parameters
        ----------
        preprocessed_data: Dict(Dict(pd.DataFrames), Dict(pd.DataFrames))
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        extracted_features: Dict(Dict(pd.DataFrames), Dict(pd.DataFrames))
        """

        extracted_features = {}
        extracted_features['io_current'] = preprocessed_data['io_current']['2020'].columns.tolist()
        extracted_features['io_old'] = preprocessed_data['io_old']['1950'].columns.tolist()
        extracted_features['io_new'] = preprocessed_data['io_new']['1980'].columns.tolist()
        extracted_features['supply'] = preprocessed_data['supply']['2020'].columns.tolist()
        extracted_features['make_old'] = preprocessed_data['make_old']['1950'].columns.tolist()
        extracted_features['make_new'] = preprocessed_data['make_new']['1980'].columns.tolist()

        return extracted_features

    def get_judgement_calls_dictionary(self):
        """Return dictionary of keyword arguments for each function in the dataset class.
        Each key should be a string with the name of the arg.
        Each value should be a list of values, with the default value coming first.

        Example
        -------
        return {
            'clean_data': {},
            'preprocess_data': {
                'imputation_strategy': ['mean', 'median'],  # first value is default
            },
            'extract_features': {},
        }
        """
        return {'clean_data'      : {},
                'preprocess_data' : {
                    'impute_missing': [True, False]
                },
                'extract_features': {}
                }

    def get_data(self, data_path: str = 'data/') -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Runs all the processing and returns the data, split into train, validation,
        and test sets.

        Params
        ------
        data_path: str, optional
            Path to all data

        Returns
        -------
        df_train
        df_tune
        df_test
        """

        np.random.seed(0)
        random.seed(0)
        CACHE_PATH = oj(data_path, 'joblib_cache')
        cache = Memory(CACHE_PATH, verbose=0).cache
        kwargs = self.get_judgement_calls_dictionary()
        default_kwargs = {}

        for key in kwargs.keys():
            func_kwargs = kwargs[key]
            default_kwargs[key] = {k: func_kwargs[k][0]  # first arg in each list is default
                                   for k in func_kwargs.keys()}
        print('kwargs', default_kwargs)

        cleaned_data = cache(self.clean_data)(data_path=data_path, **default_kwargs['clean_data'])
        preprocessed_data = cache(self.preprocess_data)(cleaned_data, **default_kwargs['preprocess_data'])

        # FIXME: extracted_features are not really the end result of the data preprocessing pipeline
        extracted_features = cache(self.extract_features)(preprocessed_data, **default_kwargs['extract_features'])

        # df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
        # return df_train, df_tune, df_test

        return preprocessed_data

if __name__ == "__main__":
    df = Dataset().get_data()

    d = df["io_current"]["1997"]