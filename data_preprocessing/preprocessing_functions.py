import pandas as pd
# import requests

# from fcd_torch import FCD
# import torch

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

import numpy as np
# import time
# import GPUtil

# import matplotlib.pyplot as plt

def merge_conditions(data, metadata, col_to_insert_before='Label'):
    """
    Merges temperature and pressure conditions from metadata into the data DataFrame.
    The 'index' column in data is used to match with 'level_0' in metadata.
    The resulting DataFrame will have 'TemperatureKelvin' and 'PressureBar' columns
    added before the 'Label' column.
    Parameters:
    data (pd.DataFrame): DataFrame containing the spectral data with an 'index' column.
    metadata (pd.DataFrame): DataFrame containing metadata with 'level_0', 'TemperatureKelvin', and 'PressureBar' columns.
    Returns:
    pd.DataFrame: DataFrame with merged conditions, including 'TemperatureKelvin' and 'PressureBar'.            
    """
    data_with_conditions = data.merge(
        metadata[['level_0', 'TemperatureKelvin', 'PressureBar']],
        left_on='index',
        right_on='level_0',
        how='left'
    )
    # Reorder columns to move TemperatureKelvin and PressureBar before Label
    cols = list(data_with_conditions.columns)
    label_index = cols.index(col_to_insert_before)
    cols.insert(label_index, cols.pop(cols.index('TemperatureKelvin')))
    cols.insert(label_index + 1, cols.pop(cols.index('PressureBar')))
    data_with_conditions = data_with_conditions[cols]
    # Only drop 'level_0' if it exists in the columns
    if 'level_0' in data_with_conditions.columns:
        data_with_conditions.drop(columns=['level_0'], inplace=True)
    return data_with_conditions

def reformat_spectra_df(df):
    """
    Transforms the 'Spectrum' column of a DataFrame into a new DataFrame where
    the 'SMILES' column is used as column headers and the formatted 'Spectrum'
    column becomes the data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'SMILES' and 'Spectrum' columns.

    Returns:
    pd.DataFrame: Transformed DataFrame.
    """
    transformed_data = {}

    for _, row in df.iterrows():
        smiles = row['SMILES']
        spectrum = row['Spectrum'].split(' ')
        indices = [round(float(pair.split(':')[0])) for pair in spectrum]
        values = [float(pair.split(':')[1]) for pair in spectrum]

        max_index = max(indices)
        spectrum_df = pd.DataFrame(np.zeros((max_index + 1, 1)), columns=[smiles])
        for idx, val in zip(indices, values):
            spectrum_df.at[idx, smiles] = val

        transformed_data[smiles] = spectrum_df[smiles]

    return pd.concat(transformed_data.values(), axis=1)

def load_data(file_path):
    if file_path.endswith('.feather'):
        data = pd.read_feather(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .feather or .csv file.")
    return data

def scale_reactant_ion_peak(data, scaling_factor=.1, rip_start_col='184', rip_stop_col='300'):
    """
    Scale reactant ion peak by a given factor.
    
    Parameters:
    scaling_factor (float): Factor by which to scale the reactant ion peak.
    rip_start_col (int): Start column name for the reactant ion peak.
    rip_stop_col (int): Stop column name for the reactant ion peak.
    
    Returns:
    pd.DataFrame: DataFrame containing the scaled data.
    """

    for spec_type in ['p_', 'n_']:
        rip_start_idx = data.columns.get_loc(f'{spec_type}{rip_start_col}')
        rip_stop_idx = data.columns.get_loc(f'{spec_type}{rip_stop_col}')
        # Multiply all intensity values between rip_start_idx and rip_stop_idx by scaling_factor
        data.iloc[:, rip_start_idx:rip_stop_idx] *= scaling_factor
    
    return data

def create_condition_dfs(metadata, spectra, condition, condition_cutoff):
    low_condition_meta = metadata[metadata[condition] < condition_cutoff]
    low_condition_indices = low_condition_meta['level_0']
    low_condition_spectra = spectra[spectra['index'].isin(low_condition_indices)]
    high_condition_meta = metadata[metadata[condition] >= condition_cutoff]
    high_condition_indices = high_condition_meta['level_0']
    high_condition_spectra = spectra[spectra['index'].isin(high_condition_indices)]
    return low_condition_meta, low_condition_spectra, high_condition_meta, high_condition_spectra

def load_data_save_condition_dfs(
    meta_file_pt1, meta_file_pt2, spectra_file_pt1, spectra_file_pt2, 
    save_file_pt1, save_file_pt2, condition, condition_cutoff
    ):
    for split_type in ['train', 'val', 'test']:
        file_path = meta_file_pt1 + split_type + meta_file_pt2
        split_meta = pd.read_csv(file_path)
        file_path = spectra_file_pt1 + split_type + spectra_file_pt2
        split_spectra = pd.read_feather(file_path)
        # print(split_type.upper(), 'value counts:')
        # print(split_spectra['Label'].value_counts())
        _, split_spectra_low_condition, _, split_spectra_high_condition = create_condition_dfs(split_meta, split_spectra, condition, condition_cutoff)
        split_spectra_high_condition.to_csv(save_file_pt1 + split_type + '_' + save_file_pt2 + '_high_' + condition + '.csv', index=False)
        # print(split_type.upper(), f'low {condition} value counts:')
        # print(split_spectra_low_condition['Label'].value_counts())
        # print(split_type.upper(), f'high {condition} value counts:')
        # print(split_spectra_high_condition['Label'].value_counts())
        # print('---------------------------------')
        # print('---------------------------------')
        split_spectra_low_condition.to_csv(save_file_pt1 + split_type + '_' + save_file_pt2 + '_low_' + condition + '.csv', index=False)
        del split_spectra, split_meta, split_spectra_high_condition, split_spectra_low_condition