# General script to run generator model
#%%
# Load Packages and Files:
import pandas as pd
val_embeddings_file_path = 'CARL/val_carls_one_per_spec.feather'#../data/encoder_embedding_predictions/val_embeddings.csv'
val_embs = pd.read_feather(val_embeddings_file_path)
val_embs.head()
#%%

# import numpy as np
# import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
# import os
import importlib
import functions as f
# import time
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_preprocessing'))
sys.path.append(parent_dir)
import preprocessing_functions as ppf
#%%
# Reload the functions module after updates
importlib.reload(f)

# generate_synthetic_data = None
generate_synthetic_data = 'y'
train_model = 'yes'
# best_hyperparameters = {'batch_size':16}

generator_load_path = None

#%%
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/run_generator.py'
target_type = 'CARL'
architecture = 'universal_generator'
model_type = 'ConditionGenerator'
loss = 'MSELoss'
criterion = nn.MSELoss()
input_type = 'carl_embeddings'
early_stopping_threshold = 10
num_plots = 5
# scaling_string = '25'
scaling_factor = .25

start_idx = 2
stop_idx = -9
device = f.set_up_gpu()

model_hyperparams = {
    'batch_size':[32],#, 32],
    'epochs': [10],
    'learning_rate':[.001]#, .001],
    }

# Comment out all wandb lines and related arguments
config = {
    # 'wandb_entity': 'catemerfeld',
    # 'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss':loss,
    'input': input_type,
    'target': target_type,
    'early stopping threshold':early_stopping_threshold
}
metadata = pd.read_feather('../../../../../scratch/cmdunham/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')

generator_save_path_pt_1 = f'trained_models/{target_type}/'
generator_save_path_pt_2 = f'{architecture}.pth'
generator_save_path = f'trained_models/{target_type}/{architecture}.pth'
synthetic_data_save_path_pt_1 = f'../../scratch/synthetic_data/{target_type}/{architecture}/'
synthetic_data_save_path_pt_2 = 'synthetic_test_spectra.feather'
train_file_path = 'CARL/train_carls_one_per_spec.feather'
# train_file_path = '../../scratch/train_data.feather'
train_embeddings_file_path = 'CARL/train_carls_one_per_spec.feather' # '../data/encoder_embedding_predictions/train_embeddings.csv'
# train_file_path = f'../../scratch/PHIL/train_phils_scaled_to_{scaling_string}_pct.csv'
# train_embeddings_file_path = f'../../scratch/PHIL/train_embedding_preds_scaled_to_{scaling_string}_pct.feather'

val_file_path = 'CARL/val_carls_one_per_spec.feather'
# val_file_path = '../../scratch/val_data.feather'
val_embeddings_file_path = 'CARL/val_carls_one_per_spec.feather' # '../data/encoder_embedding_predictions/val_embeddings.csv'
# val_file_path = f'../../scratch/PHIL/val_phils_scaled_to_{scaling_string}_pct.csv'
# val_embeddings_file_path = f'../../scratch/PHIL/val_embedding_preds_scaled_to_{scaling_string}_pct.feather'

test_file_path = 'CARL/test_carls_one_per_spec.feather'
# test_file_path = '../../scratch/test_data.feather'
test_embeddings_file_path = 'CARL/test_carls_one_per_spec.feather' # '../data/encoder_embedding_predictions/test_embeddings.csv'
# test_file_path = f'../../scratch/PHIL/test_phils_scaled_to_{scaling_string}_pct.csv'
# test_embeddings_file_path = f'../../scratch/PHIL/test_embedding_preds_scaled_to_{scaling_string}_pct.feather'
test_avg_bkg_file_path = None  # '../../scratch/test_avg_bkg.csv' (not present, so set to None)

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

if architecture == 'group_generator':
    chem_groups = [['DMMP', 'TEPO'], ['DEM', 'DPM', 'DEB'], ['DtBP', 'MES']]
elif architecture == 'universal_generator':
    chem_groups = ['all chemicals']

# Lines that only apply to group generators marked with #####
###
for group in chem_groups:
    if chem_groups[0] != 'all chemicals':
        group_file_path = '_'.join(group)
        generator_save_path = '_'.join([generator_save_path_pt_1,group_file_path,generator_save_path_pt_2])
        wandb_kwargs['group']=group
        #####

    # Loading data needs to be redone for each group because keeping the entire dataset in memory + the group subset is too much
    train_data = ppf.load_data(train_file_path)
    # print(train_data.head())
    train_embeddings_df = ppf.load_data(train_embeddings_file_path)

    train_embeddings_with_conditions = ppf.merge_conditions(
        train_embeddings_df, metadata, col_to_insert_before='DEB',
    )
    # train_embeddings_df.merge(
    #     metadata[['level_0', 'TemperatureKelvin', 'PressureBar']],
    #     left_on='index',
    #     right_on='level_0',
    #     how='left'
    # )
    # cols = list(train_embeddings_with_conditions.columns)
    # label_index = cols.index('DEB')
    # cols.insert(label_index, cols.pop(cols.index('TemperatureKelvin')))
    # cols.insert(label_index + 1, cols.pop(cols.index('PressureBar')))
    # train_embeddings_with_conditions = train_embeddings_with_conditions[cols]
    # train_embeddings_with_conditions.drop(columns=['level_0'], inplace=True)

    #####
    if chem_groups[0] != 'all chemicals':
        train_data = train_data[train_data[group].any(axis=1)] 
        train_embeddings_df = train_embeddings_df[train_embeddings_df[group].any(axis=1)] 
        train_embeddings_with_conditions = train_embeddings_with_conditions[train_embeddings_with_conditions[group].any(axis=1)]

    #####
    
    x_train, y_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors_for_generator(
        train_data, train_embeddings_with_conditions, device, start_idx=start_idx, stop_idx=stop_idx)

    del train_data, train_embeddings_df, train_embeddings_with_conditions

    #%%
    val_data = ppf.load_data(val_file_path)
    val_embeddings_df = ppf.load_data(val_embeddings_file_path)

    val_embeddings_with_conditions = ppf.merge_conditions(
        val_embeddings_df, metadata, col_to_insert_before='DEB',
    )
    # #####
    if chem_groups[0] != 'all chemicals':
        val_data = val_data[val_data[group].any(axis=1)] 
        val_embeddings_df = val_embeddings_df[val_embeddings_df[group].any(axis=1)]
        val_embeddings_with_conditions = val_embeddings_with_conditions[val_embeddings_with_conditions[group].any(axis=1)]
    # #####

    x_val, y_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors_for_generator(
        val_data, val_embeddings_with_conditions, device, start_idx=start_idx, stop_idx=stop_idx)
    del val_data, val_embeddings_df, val_embeddings_with_conditions
    # %%
    test_data = ppf.load_data(test_file_path)
    test_cols = test_data.columns[start_idx:stop_idx]
    test_embeddings_df = ppf.load_data(test_embeddings_file_path)

    # #####
    if chem_groups[0] != 'all chemicals':
        test_data = test_data[test_data[group].any(axis=1)]
        test_embeddings_df = test_embeddings_df[test_embeddings_df[group].any(axis=1)] 
    # #####

    x_test, y_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors_for_generator(
        test_data, test_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

    del test_data, test_embeddings_df

    test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)
    #%%
    if train_model == 'yes':
        train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
        val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)

        best_hyperparameters = f.train_generator(
            train_data, val_data, test_data, device, config,
            None,  # wandb_kwargs (not used, pass None)
            model_hyperparams, sorted_chem_names,
            generator_save_path,  # generator_path argument
            save_plots_to_wandb=False,  # set to False
            early_stop_threshold=early_stopping_threshold, 
            num_plots=num_plots, model_type=model_type, pretrained_model_path=generator_load_path,
            carl_or_spec=target_type
        )
        #%%
    if generate_synthetic_data == None:
        generate_synthetic_data = f.get_input()

    if generate_synthetic_data == 'y':
        if chem_groups[0] != 'all chemicals':
            print(f'Generating data for group: {group}')
            group_file_path = '_'.join(group)
            generator_path = '_'.join([generator_save_path_pt_1,group_file_path,generator_save_path_pt_2])
            preds_save_path = '_'.join([synthetic_data_save_path_pt_1,group_file_path,synthetic_data_save_path_pt_2])
        if chem_groups[0] == 'all chemicals':
            print('Generating synthetic data...')
            generator_path = generator_save_path
            preds_save_path = '_'.join([synthetic_data_save_path_pt_1,synthetic_data_save_path_pt_2])
            
        batch_size = best_hyperparameters['batch_size']

        # ####
        # # for group generators
        # for group in chem_groups:
        #     print(f'Generating data for group: {group}')
        #     group_file_path = '_'.join(group)
        #     generator_path = '_'.join([generator_save_path_pt_1,group_file_path,generator_save_path_pt_2])
        #     preds_save_path = '_'.join([synthetic_data_save_path_pt_1,group_file_path,synthetic_data_save_path_pt_2])
        # #####


        # # Everything below indented for group gens
        # test_data = ppf.load_data(test_file_path)
        # test_embeddings_df = ppf.load_data(test_embeddings_file_path)
        # test_cols = test_data.columns[start_idx:stop_idx]

        # # #####
        # # test_data = test_data[test_data[group].any(axis=1)]
        # # test_embeddings_df = test_embeddings_df[test_embeddings_df[group].any(axis=1)] 
        # # #####

        # x_test, y_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors_for_generator(
        #     test_data, test_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

        # del test_data, test_embeddings_df

        # test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)
        test_dataset = DataLoader(test_data, batch_size)

        model = f.load_model(generator_path, device=device)#, weights_only=False)
        test_preds, test_chem_name_encodings, _, test_indices = f.predict_embeddings(test_dataset, model, device, criterion)


        ##############
        if target_type == 'CARL' and test_avg_bkg_file_path is not None:
            test_avg_bkg = pd.read_csv(test_avg_bkg_file_path)
            test_avg_bkg.drop(columns=['Unnamed: 0'], inplace=True)

            preds_list = [pred for pred_list in test_preds for pred in pred_list]

            synthetic_spectra = []

            for pred in preds_list:
                synthetic_spec = pred + test_avg_bkg
                synthetic_spectra.append(synthetic_spec.values.flatten())

        else: 
            synthetic_spectra = [pred for pred_list in test_preds for pred in pred_list]

        indices_list = [ind for ind_list in test_indices for ind in ind_list]
        # Create list of chemical names for generated spectra
        test_chem_name_encodings_list = [enc for enc_list in test_chem_name_encodings for enc in enc_list]
        test_labels = [sorted_chem_names[list(enc).index(1)] for enc in test_chem_name_encodings_list]

        synthetic_spectra_df = pd.DataFrame(synthetic_spectra, columns=test_cols)
        synthetic_spectra_df['Label'] = test_labels
        synthetic_spectra_df['index'] = indices_list

        if target_type == 'PHIL':
            synthetic_spectra_df = ppf.scale_reactant_ion_peak(
                synthetic_spectra_df, scaling_factor=1/scaling_factor
            )
        
        synthetic_spectra_df.to_feather(preds_save_path)#, index=False)
    else:
        print('Skipping synthetic data generation...')
