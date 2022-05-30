"""helper.py: Global helper functions"""


import numpy as np
import pandas as pd


def rename_columns(dict, is_io, is_historical):
    """
    :param: dict: dictionary of pandas dataframes
    :param: is_io: boolean - if true, io data is fed in
    :param: is_recent - should be true if data comes after 1962
    # outputs dictionary of dataframes with columns renamed and empty rows removed
    """
    for key in list(dict.keys()):
        if is_historical:
            dict[key].columns = list(dict[key].iloc[4, :])
            if is_io:
                dict[key] = dict[key].iloc[6:, 1:]
            else:
                dict[key] = dict[key].iloc[6:, :]
        else:
            dict[key].columns = list(dict[key].iloc[5, :])
            if is_io:
                dict[key].columns = list(dict[key].iloc[5, :])
                # drop those first 5 columns as they have no information
                dict[key] = dict[key].iloc[6:-4, 1:]
            else:
                dict[key] = dict[key].iloc[6:, 1:]
    return dict


def replace_zeros(dict, is_input):
    """
    :param: dict: dictionary of pandas dataframes
    :param: is_input: boolean - if true, data is input dictionary
    # outputs dictionary of dataframes with zeros removed from the industry columns
    """
    for key in list(dict.keys()):
        if is_input:
            # drop those first 5 columns as they have no information
            dict[key].iloc[:-6] = dict[key].iloc[:-6].replace(np.nan, 0)
            dict[key].iloc[-5:-2] = dict[key].iloc[-5:-2].replace(np.nan, 0)
        else:
            dict[key].iloc[:-12] = dict[key].iloc[:-12].replace(np.nan, 0)
    return dict


def replace_dots(dict):
    """
    Replace '...' to np.NaN
    :param: dict - dictionary of dataframes to replace values with
    :return: dict with np.NaNs instead of '...'
    """
    return {k: v.replace('...', np.nan) for k, v in dict.items()}


def write_to_separate_csvs(py_path):
    """
    Takes the dictionary of dictionaries of dataframes and returns separate
    csvs by type
    :param: py_path - path to base directory - i.e. ~/Economic_Networks
    """
    # initiate data class and call the functions to make a dictionary of dictionaries of dataframes
    data_path = f"{str(py_path)}/data"
    Dataset = dataset.Dataset()
    data = Dataset.clean_data(data_path=data_path)
    processed_data = Dataset.preprocess_data(cleaned_data=data)

    # grabbing list of data types and valid years
    data_types = list(processed_data.keys())
    old_years = list(processed_data['io_old'].keys())[2:]
    new_years = list(processed_data['io_new'].keys())[2:]
    current_years = list(processed_data['io_current'].keys())

    # rename the ugly column names and add year and data_type flags
    for data_type in data_types:
        if data_type == 'io_current' or data_type == 'supply':
            for year in current_years:
                processed_data[data_type][year].columns = processed_data[data_type][year].columns.fillna('Name')
                # processed_data[data_type][year].columns = [col.replace(' ', '_').replace(',', '').lower() for col in processed_data[data_type][year].columns]
                processed_data[data_type][year].insert(0, 'year', year)
                processed_data[data_type][year].insert(1, 'data_type', data_type)
        elif data_type == 'io_old' or data_type == 'make_old':
            for year in old_years:
                processed_data[data_type][year].columns = processed_data[data_type][year].columns.fillna('Name')
                # processed_data[data_type][year].columns = [col.replace(' ', '_').replace(',', '').lower() for col in processed_data[data_type][year].columns]
                processed_data[data_type][year].insert(0, 'year', year)
                processed_data[data_type][year].insert(1, 'data_type', data_type)
        elif data_type == 'io_new' or data_type == 'make_new':
            for year in new_years:
                processed_data[data_type][year].columns = processed_data[data_type][year].columns.fillna('Name')
                # processed_data[data_type][year].columns = [col.replace(' ', '_').replace(',', '').lower() for col in processed_data[data_type][year].columns]
                processed_data[data_type][year].insert(0, 'year', year)
                processed_data[data_type][year].insert(1, 'data_type', data_type)

    # concatenate the dataframes by type
    for data_type in data_types:
        if data_type == 'io_current' or data_type == 'supply':
            processed_data[data_type] = pd.concat(list(processed_data[data_type].values()), axis=0)
        elif data_type == 'io_old' or data_type == 'make_old' or data_type == 'io_new' or data_type == 'make_new':
            processed_data[data_type] = pd.concat(list(processed_data[data_type].values())[2:], axis=0)

            # write the data type dataframes to csv
    for data_type in data_types:
        processed_data[data_type].to_csv(f'{data_path}/processed/{data_type}.csv')
