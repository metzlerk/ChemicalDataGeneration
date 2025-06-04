#%%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import itertools
import GPUtil
from collections import Counter, OrderedDict
import dask.dataframe as dd
import os
# import random
# import psutil
import plotting_functions as pf
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from signal import signal, SIGALRM, alarm

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def load_data(file_path_parts_list, file_ending):
    """
    Load data from a file path constructed from the provided parts.
    Args:
        file_path_parts_list (list): List of strings representing parts of the file path.   
        file_ending (str): The file ending of the synthetic data file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if len(file_path_parts_list) > 1:
        file_path = '_'.join(file_path_parts_list)
    else:
        file_path = file_path_parts_list[0]
        
    if file_ending == 'feather':
        data = pd.read_feather(file_path)
    elif file_ending == 'csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file ending: {file_ending}")
    return data
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class Timeout(Exception):
    ...

def get_input(time_limit=120):
    def handler(*_):
        raise Timeout
    old_handler = signal(SIGALRM, handler)
    alarm(time_limit)
    try:
        alarm(0)
        return input(f'Generate predictions using best trained model? (y/n) ')
    except Timeout:
        print('Time limit exceeded. Exiting script.')
    finally:
        signal(SIGALRM, old_handler)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def get_onehot_labels(data, one_hot_columns_idx_list):
    one_hot_columns = data.columns[one_hot_columns_idx_list[0]:one_hot_columns_idx_list[1]]
    one_hot_labels = data[one_hot_columns].idxmax(axis=1)
    return one_hot_labels

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# def oversample_condition_data(data_to_oversample, data_to_match):
#     if 'Label' not in data_to_match.columns:
#         # Extract the one-hot encoded columns
#         one_hot_columns = data_to_match.columns[-8:]

#         # Convert one-hot encodings to 'Label' column
#         data_to_match['Label'] = data_to_match[one_hot_columns].idxmax(axis=1)

#     # Not all chems have data for both conditions. 
#     # Drop rows in data_to_match if their 'Label' is not in data_to_oversample['Label']
#     labels_to_keep = data_to_oversample['Label'].unique()
#     data_to_match = data_to_match[data_to_match['Label'].isin(labels_to_keep)]

#     class_counts = data_to_match['Label'].value_counts()

#     X = data_to_oversample.drop(columns=['Label'], axis=1)
#     y = data_to_oversample['Label']

#     ros = RandomOverSampler(sampling_strategy=class_counts.to_dict(), random_state=42)
#     X_resampled, y_resampled = ros.fit_resample(X, y)

#     upsampled_data = pd.DataFrame(X_resampled, columns=X.columns).copy()
#     upsampled_data['Label'] = y_resampled

#     data_to_match = data_to_match.sort_values(by=['Label'])
#     upsampled_data = upsampled_data.sort_values(by=['Label'])

#     data_to_match.drop(columns=['Label'], inplace=True)
#     return upsampled_data, data_to_match

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def resample_condition_data(data_to_resample, data_to_match, sampling_technique='over'):
    if 'Label' not in data_to_match.columns:
        data_to_match['Label'] = get_onehot_labels(data_to_match, one_hot_columns_idx_list=[-8, None])
    
    # Not all chems have data for both conditions. 
    # Drop rows of each dataframe if their 'Label' is not in the other dataframe
    data_to_resample = data_to_resample[data_to_resample['Label'].isin(data_to_match['Label'].unique())]
    data_to_match = data_to_match[data_to_match['Label'].isin(data_to_resample['Label'].unique())]

    class_counts = data_to_match['Label'].value_counts()

    X = data_to_resample.drop(columns=['Label'], axis=1)
    y = data_to_resample['Label']

    if sampling_technique == 'over':
        sampler = RandomOverSampler(sampling_strategy=class_counts.to_dict(), random_state=42)
    elif sampling_technique == 'under':
        sampler = RandomUnderSampler(sampling_strategy=class_counts.to_dict(), random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    resampled_data = pd.DataFrame(X_resampled, columns=X.columns).copy()
    resampled_data['Label'] = y_resampled

    data_to_match = data_to_match.sort_values(by=['Label'])
    resampled_data = resampled_data.sort_values(by=['Label'])

    data_to_match.drop(columns=['Label'], inplace=True)
    return resampled_data, data_to_match
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def format_preds_df(input_indices, predicted_embeddings, output_name_encodings, sorted_chem_names):
    input_indices = [idx for idx_list in input_indices for idx in idx_list]
    predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
    output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
    preds_df = pd.DataFrame(predicted_embeddings)
    preds_df.insert(0, 'index', input_indices)
    name_encodings_df = pd.DataFrame(output_name_encodings)
    name_encodings_df.columns = sorted_chem_names
    preds_df = pd.concat([preds_df, name_encodings_df], axis=1)
    preds_df.columns = preds_df.columns.astype(str)
    return preds_df

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def combine_embeddings(ims_embeddings_df, mass_spec_embeddings_df): 
    ims_embeddings = pd.DataFrame([emb for emb in ims_embeddings_df['Embedding Floats']][1:]).T
    mass_spec_embeddings = pd.DataFrame([emb for emb in mass_spec_embeddings_df['Embedding Floats']]).T
    cols = ims_embeddings_df.index[1:]
    ims_embeddings.columns = cols
    cols = mass_spec_embeddings_df.index
    mass_spec_embeddings.columns = cols
    all_true_embeddings = pd.concat([ims_embeddings, mass_spec_embeddings], axis=1)
    return all_true_embeddings

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def format_embedding_df(file_path, chem_list=None):
    embedding_df = pd.read_csv(file_path)
    embedding_df.set_index('Unnamed: 0', inplace=True)

    embedding_floats = []
    for chem_name in embedding_df.index:
        if chem_list is not None:
            if chem_name not in chem_list:
                # print(chem_name)
                embedding_floats.append(None)
                continue
            
        if chem_name == 'BKG':
            embedding_floats.append(None)
        else:
            embedding_float = embedding_df['embedding'][chem_name].split('[')[1]
            embedding_float = embedding_float.split(']')[0]
            embedding_float = [np.float32(num) for num in embedding_float.split(',')]
            embedding_floats.append(embedding_float)

    embedding_df['Embedding Floats'] = embedding_floats

    return embedding_df

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def run_generator(
        file_path_dict, chem, model_hyperparams, wandb_kwargs, 
        sorted_chem_names, generator_path, notebook_name, num_plots, 
        model_type='chemnet_to_ims_generator'
        ):
    
    device = set_up_gpu()
    train_embeddings_tensor, train_data_tensor, train_chem_encodings_tensor, train_indices_tensor = create_individual_chemical_dataset_tensors(
    file_path_dict['train_data_file_path'], file_path_dict['train_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )
    val_embeddings_tensor, val_data_tensor, val_chem_encodings_tensor, val_indices_tensor = create_individual_chemical_dataset_tensors(
    file_path_dict['val_data_file_path'], file_path_dict['val_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )
    test_embeddings_tensor, test_data_tensor, test_chem_encodings_tensor, test_indices_tensor = create_individual_chemical_dataset_tensors(
    file_path_dict['test_data_file_path'], file_path_dict['test_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )

    train_data = TensorDataset(train_embeddings_tensor, train_chem_encodings_tensor, train_data_tensor, train_indices_tensor)
    val_data = TensorDataset(val_embeddings_tensor, val_chem_encodings_tensor, val_data_tensor, val_indices_tensor)
    test_data = TensorDataset(test_embeddings_tensor, test_chem_encodings_tensor, test_data_tensor, test_indices_tensor)

    # remove from memory since information is now stored in train/val/test datasets
    del train_embeddings_tensor, train_chem_encodings_tensor, train_data_tensor, train_indices_tensor
    del val_embeddings_tensor, val_chem_encodings_tensor, val_data_tensor, val_indices_tensor
    del test_embeddings_tensor, test_chem_encodings_tensor, test_data_tensor, test_indices_tensor

    config = {
        'wandb_entity': 'catemerfeld',
        'wandb_project': 'ims_encoder_decoder',
        'gpu':True,
        'threads':1,
    }

    os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

    train_generator(
        train_data, val_data, test_data, device, config, 
        wandb_kwargs, model_hyperparams, sorted_chem_names, 
        generator_path, early_stop_threshold=wandb_kwargs['early stopping threshold'], 
        lr_scheduler=True, num_plots=num_plots, plot_overlap_pca=True,
        model_type=model_type
        )

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def get_class_weights(data, device):
    class_counts = dict(sorted(Counter(data['Label']).items()))
    class_weights = {cls: len(data) / count for cls, count in class_counts.items()}
    class_weights = torch.tensor(list(class_weights.values()))
    return class_weights.to(device)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


# def create_individual_chemical_dataset_tensors(carl_dataset, embedding_preds, device, chem, multiple_carls_per_spec=True):
def create_individual_chemical_dataset_tensors(carl_dataset_file_path, embedding_preds_file_path, device=None, chem=None, multiple_carls_per_spec=True):
    carl_dataset = pd.read_feather(carl_dataset_file_path)
    carl_dataset.drop('level_0', axis=1, inplace=True)
    chem_carl_dataset = carl_dataset[carl_dataset['Label'] == chem]
    # del carl_dataset
    embedding_preds = pd.read_csv(embedding_preds_file_path)
    chem_embedding_preds = embedding_preds[embedding_preds[chem] == 1.0]
    # del embedding_preds
    return create_dataset_tensors_for_generator(chem_carl_dataset, chem_embedding_preds, device, multiple_carls_per_spec)

def flatten_and_bin(predicted_embeddings_batches):
    """
    Flatten prediction batches and convert to binary format.

    This function takes a list of batches containing predicted embeddings,
    flattens them, and converts each embedding into a binary vector where 
    the index of the maximum value is set to 1, and all other indices are 0.

    Parameters:
    ----------
    predicted_embeddings_batches : list of list of torch.Tensor
        Batches of predicted embeddings, with each embedding represented as a tensor.

    Returns:
    -------
    list of list of int
        A list of binary vectors corresponding to each embedding, with a 1 at 
        the index of the maximum value.
    """
    binary_preds_list = []
    
    for batch in predicted_embeddings_batches:
        for encoding in batch:
            # Get the index of the maximum value
            max_index = torch.argmax(encoding)
            # Create a binary label with 1 in the index of the highest value's index in the encoding 0s in all other indices
            binary_pred = [0] * len(encoding)
            binary_pred[max_index] = 1
            binary_preds_list.append(binary_pred)
    
    return binary_preds_list

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# def run_with_wandb(config, **kwargs):
#     """
#     Initialize a WandB run with the given configuration.

#     This function updates the provided configuration with additional keyword 
#     arguments, initializes a WandB run, sets the number of threads for PyTorch, 
#     and determines the device (GPU or CPU) to be used for computations.

#     Parameters:
#     ----------
#     config : dict
#         Configuration dictionary containing WandB settings and other parameters.
#         Must include 'wandb_entity', 'wandb_project', and 'threads'.

#     **kwargs : keyword arguments
#         Additional configuration parameters to be added to the `config`.
#     """
#     config.update(kwargs)

#     wandb.init(entity=config['wandb_entity'],
#                project=config['wandb_project'],
#                config=config)

#     # Set the number of threads
#     torch.set_num_threads(config['threads'])

#     # Find out is there is a GPU available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if not config['gpu']:
#         device = torch.device('cpu')
#     print(f'Using device: {device}')

# # ------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------
 

# def update_wandb_kwargs(wandb_kwargs, updates):
#     """
#     Update a dictionary of WandB keyword arguments with new values.

#     Parameters:
#     ----------
#     wandb_kwargs : dict
#         The original dictionary of WandB keyword arguments to be updated.

#     updates : dict
#         A dictionary containing new values to update in `wandb_kwargs`.

#     Returns:
#     -------
#     dict
#         The updated dictionary of WandB keyword arguments.
#     """
#     for key in updates.keys():
#         wandb_kwargs[key] = updates[key]
#     return wandb_kwargs

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def train_one_epoch(
        train_dataset, device, model, criterion, optimizer, 
        epoch, combo
        ):
  """
    Train the model for one epoch on the given training dataset.

    This function performs forward and backward passes for each batch in the 
    training dataset, computes the loss, and updates the model weights. 
    It also collects predicted embeddings and name encodings at the last epoch.

    Parameters:
    ----------
    train_dataset : iterable
        An iterable dataset that yields batches of data, name encodings, true 
        embeddings, and additional information.

    device : torch.device
        The device (CPU or GPU) on which to perform the training.

    model : torch.nn.Module
        The model to be trained.

    criterion : callable
        The loss function used to compute the loss.

    optimizer : torch.optim.Optimizer
        The optimizer used to update the model weights.

    epoch : int
        The current epoch number.

    combo : dict
        A dictionary containing configuration settings, including 'epochs'.

    Returns:
    -------
    float
        The average training loss for the epoch. If it's the last epoch, 
        also returns the predicted embeddings and name encodings.

    tuple (float, list, list)
        At last epoch (either final or early stopping), returns a tuple containing the average loss, 
        a list of predicted embeddings, and a list of corresponding name encodings.

  """
  epoch_training_loss = 0

  predicted_embeddings = []
  output_name_encodings = []

  for batch, name_encodings, true_embeddings, _ in train_dataset:
    # move inputs to device
    batch = batch.to(device)
    name_encodings = name_encodings.to(device)
    true_embeddings = true_embeddings.to(device)

    # backprapogation
    optimizer.zero_grad()
    
    batch_predicted_embeddings = model(batch)

    if isinstance(model, IMStoOneHotEncoder):
        class_indices = torch.argmax(true_embeddings, dim=1)
        loss = criterion(batch_predicted_embeddings, class_indices)
    else:
        loss = criterion(batch_predicted_embeddings, true_embeddings)
    # accumulate epoch training loss
    epoch_training_loss += loss.item()

    loss.backward()
    optimizer.step()

    # at last epoch store output embeddings and corresponding labels to output list
    if epoch == combo['epochs']:
      output_name_encodings.append(name_encodings)
      predicted_embeddings.append(batch_predicted_embeddings)

  # divide by number of batches to calculate average loss
  average_loss = epoch_training_loss/len(train_dataset)
  if epoch == combo['epochs']:
    return average_loss, predicted_embeddings, output_name_encodings
  else:
    return average_loss
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# def format_embedding_predictions(predicted_embeddings, input_indices, output_name_encodings, sorted_chem_names):
#     # input_indices = [idx for idx_list in input_indices for idx in idx_list]
#     input_indices = [int(idx) for idx in input_indices]
#     predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
#     output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
#     preds_df = pd.DataFrame(predicted_embeddings)
#     preds_df.insert(0, 'index', input_indices)
#     name_encodings_df = pd.DataFrame(output_name_encodings)
#     name_encodings_df.columns = sorted_chem_names
#     preds_df = pd.concat([preds_df, name_encodings_df], axis=1)
#     return preds_df

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 

def predict_embeddings(dataset, model, device, criterion, reparameterization=False):
    """
    Generate predicted embeddings and compute average loss on the given dataset.

    This function evaluates the model on the provided dataset, computes the predicted 
    embeddings, and calculates the average loss by comparing predictions to true embeddings.

    Parameters:
    ----------
    dataset : iterable
        An iterable dataset that yields batches of data, name encodings, true embeddings, 
        and spectra indices.

    model : torch.nn.Module
        The model to be evaluated.

    device : torch.device
        The device (CPU or GPU) on which to perform the evaluations.

    criterion : callable
        The loss function used to compute the loss between predicted and true embeddings.

    Returns:
    -------
    tuple
        A tuple containing:
        - predicted_embeddings (list): List of predicted embeddings for each batch.
        - output_name_encodings (list): List of name encodings for the predicted embeddings.
        - average_loss (float): The average loss over all batches.
        - input_spectra_indices (list): List of spectra indices corresponding to the input data.
    """
    total_loss = 0

    model.eval() # Set model to evaluation mode
    predicted_embeddings = []
    output_name_encodings = []
    input_spectra_indices = []

    # print('Predicting embeddings...')
    with torch.no_grad():
        for batch, name_encodings, true_embeddings, spectra_indices in dataset:
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)

            if reparameterization:
                batch_predicted_embeddings = model(batch, reparameterization)
            else:
                batch_predicted_embeddings = model(batch)
            predicted_embeddings.append(batch_predicted_embeddings.to('cpu').detach().numpy())
            output_name_encodings.append(name_encodings.to('cpu').detach().numpy())
            input_spectra_indices.append(spectra_indices.to('cpu').detach().numpy())

            # print(batch_predicted_embeddings.shape, true_embeddings.shape)

            loss = criterion(batch_predicted_embeddings, true_embeddings)
            # accumulate loss
            total_loss += loss.item()

    # divide by number of batches to calculate average loss
    average_loss = total_loss/len(dataset)
    return predicted_embeddings, output_name_encodings, average_loss, input_spectra_indices

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(1676,1548),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1548,1420),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1420, 1292),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1292, 1164),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1164, 1036),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1036, 908),
      nn.LeakyReLU(inplace=True),
      nn.Linear(908, 780),
      nn.LeakyReLU(inplace=True),
      nn.Linear(780, 652),
      nn.LeakyReLU(inplace=True),
    )

    # self.final_relu = 
    self.final_linear = nn.Linear(652, 512)

    self.mean_layer = nn.Linear(652, 512)
    self.logvar_layer = nn.Linear(652, 512)

  def reparameterize(self, mean, log_var):
    eps = torch.randn_like(log_var)
    z = mean + log_var * eps
    return z

  def forward(self, x, reparameterization=False):
    x = self.encoder(x)
    
    # Do reparameterization if desired, otherwise run final encoder layer
    if reparameterization:
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterize(mean, logvar)
        return z
    else:
        x = self.final_linear(x)
        return x
#%%
class ConditionEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(1678,1548),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1548,1420),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1420, 1292),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1292, 1164),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1164, 1036),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1036, 908),
      nn.LeakyReLU(inplace=True),
      nn.Linear(908, 780),
      nn.LeakyReLU(inplace=True),
      nn.Linear(780, 652),
      nn.LeakyReLU(inplace=True),
      nn.Linear(652, 514)
    )

  def forward(self, x):
    x = self.encoder(x)
    return x

#%%
class IMStoOneHotEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(1676,1491),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1491,1306),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1306, 1121),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1121, 936),
      nn.LeakyReLU(inplace=True),
      nn.Linear(936, 751),
      nn.LeakyReLU(inplace=True),
      nn.Linear(751, 566),
      nn.LeakyReLU(inplace=True),
      nn.Linear(566, 381),
      nn.LeakyReLU(inplace=True),
      nn.Linear(381, 196),
      nn.LeakyReLU(inplace=True),
      nn.Linear(196, 8),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x

#%%
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class OneHottoChemNetEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(8, 72),
      nn.LeakyReLU(inplace=True),
      nn.Linear(72, 136),
      nn.LeakyReLU(inplace=True),
      nn.Linear(136, 200),
      nn.LeakyReLU(inplace=True),
      nn.Linear(200, 264),
      nn.LeakyReLU(inplace=True),
      nn.Linear(264, 328),
      nn.LeakyReLU(inplace=True),
      nn.Linear(328, 392),
      nn.LeakyReLU(inplace=True),
      nn.Linear(392, 456),
      nn.LeakyReLU(inplace=True),
      nn.Linear(456, 512),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x
#%%


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def train_model(
        model_type, train_data, val_data, test_data, device, config, wandb_kwargs, 
        all_embeddings_df, ims_embeddings_df, model_hyperparams, sorted_chem_names, 
        encoder_path, save_emb_pca_to_wandb = True, early_stop_threshold=10, 
        input_type='IMS', embedding_type='ChemNet', show_wandb_run_name=True, 
        lr_scheduler = False, patience=5, class_weights=None
        ):
    
    """
    Train a model with specified hyperparameters and log results using Weights & Biases.

    Parameters:
    ----------
    model_type : str
        The type of model to be trained (e.g., 'Encoder').

    train_data : Dataset
        The dataset used for training the model.

    val_data : Dataset
        The dataset used for validating the model during training.

    test_data : Dataset
        The dataset used for evaluating the model after training.

    device : torch.device
        The device (CPU or GPU) on which to perform the training.

    config : dict
        Configuration settings for the training process.

    wandb_kwargs : dict
        Arguments for logging to Weights & Biases.

    all_embeddings_df : pd.DataFrame
        DataFrame containing all embeddings for chemicals.

    ims_embeddings_df : pd.DataFrame
        DataFrame containing IMS (ion mobility spectrometry) embeddings.

    model_hyperparams : dict
        Dictionary of hyperparameters for the model, with keys as parameter names and values as lists of options.

    sorted_chem_names : list of str
        List of sorted chemical names corresponding to the embeddings.

    encoder_path : str
        File path to save the best model state.

    save_emb_pca_to_wandb : bool, optional
        If True, saves PCA plots of embeddings to Weights & Biases. Default is True.

    early_stop_threshold : int, optional
        Number of epochs without improvement in validation loss before stopping training. Default is 10.

    input_type : str, optional
        The type of input being used (IMS, Carl, MNIST). Default is 'IMS'.

    embedding_type : str, optional
        The type of embedding being used (ChemNet, OneHot). Default is 'ChemNet'.

    show_wandb_run_name : bool, optional
        If True, displays the current WandB run name on the plot. Default is True.

    Returns:
    -------
    dict
        The best hyperparameters found during training.
    """

    # loss to compare for each model. Starting at infinity so it will be replaced by first model's first epoch loss 
    lowest_val_loss = np.inf

    keys = model_hyperparams.keys()
    values = model_hyperparams.values()

    # Generate all parameter combinations from model_config using itertools.product
    combinations = itertools.product(*values)

    # Iterate through each parameter combination and run model 
    for combo in combinations:
        # creating different var for model loss to use for early stopping
        lowest_val_model_loss = np.inf
        
        if model_type == 'Encoder':
            model = Encoder().to(device)
            criterion = nn.MSELoss()
        
        elif model_type == 'ConditionEncoder':
            model = ConditionEncoder().to(device)
            criterion = nn.MSELoss()

        elif model_type == 'Generator':
            model = Generator().to(device)
            criterion = nn.MSELoss()
        
        elif model_type == 'ConditionGenerator':
            model = ConditionGenerator().to(device)
            criterion = nn.MSELoss()

        elif model_type == 'IMStoOneHotEncoder':
            model = IMStoOneHotEncoder().to(device)
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        elif model_type == 'OneHottoChemNetEncoder':
            model = OneHottoChemNetEncoder().to(device)
            criterion = nn.MSELoss()

        epochs_without_validation_improvement = 0
        combo = dict(zip(keys, combo))

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr = combo['learning_rate'])

        final_lr = combo['learning_rate']

        if lr_scheduler:
            # Initialize the learning rate scheduler with patience of 5 epochs 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

        # wandb_kwargs = update_wandb_kwargs(wandb_kwargs, combo)

        # run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(1, combo['epochs']+1):
            if epochs_without_validation_improvement < early_stop_threshold:
                model.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if epoch == combo['epochs']:
                    average_loss, _, _ = train_one_epoch(
                    train_dataset, device, model, criterion, 
                    optimizer, epoch, combo
                    )
                    # wandb.log({'Learning Rate at Final Epoch':final_lr})
                    # save output pca to weights and biases
                    if save_emb_pca_to_wandb:
                        # plot_pca gets predictions from trained model and plots them
                        pf.plot_pca(
                            train_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df, 
                            ims_embeddings_df, 'Train', input_type, embedding_type, show_wandb_run_name
                            )
                        pf.plot_pca(
                            test_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df,
                            ims_embeddings_df, 'Test', input_type, embedding_type, show_wandb_run_name
                            )
                else:
                    average_loss = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )

                epoch_val_loss = 0  
                # evaluate model on validation data
                model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch, val_name_encodings, val_true_embeddings, _ in val_dataset:
                        val_batch = val_batch.to(device)
                        val_name_encodings = val_name_encodings.to(device)
                        val_true_embeddings = val_true_embeddings.to(device)

                        val_batch_predicted_embeddings = model(val_batch)

                        val_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)
                        # accumulate epoch validation loss
                        epoch_val_loss += val_loss.item()

                # divide by number of batches to calculate average loss
                val_average_loss = epoch_val_loss/len(val_dataset)

                if lr_scheduler:
                    scheduler.step(val_average_loss)  # Pass the validation loss to the scheduler
                    # get the new learning rate (to give to wandb)
                    final_lr = optimizer.param_groups[0]['lr']

                if val_average_loss < lowest_val_model_loss:
                    # check if val loss is improving for this model
                    epochs_without_validation_improvement = 0
                    lowest_val_model_loss = val_average_loss

                    if val_average_loss < lowest_val_loss:
                        # if current epoch of current model is best performing (of all epochs and models so far), save model 
                        torch.save(model, encoder_path)
                        print(f'Saved best model at epoch {epoch}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                    else:
                        print(f'Model best validation loss at {epoch}')
                
                else:
                    epochs_without_validation_improvement += 1

                # log losses to wandb
                # if model_type == 'Encoder':
                # wandb.log({f"{model_type} Training Loss": average_loss, f"{model_type} Validation Loss": val_average_loss})
                # elif model_type == 'Generator':
                #     wandb.log({"Generator Training Loss": average_loss, "Generator Validation Loss": val_average_loss})

                if epoch % 10 == 0 or epoch == 0:
                    print('Epoch[{}/{}]:'.format(epoch, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch}.')
                # wandb.log({'Early Stopping Ecoch':epoch})
                # wandb.log({'Learning Rate at Final Epoch':final_lr})
                pf.plot_pca(
                    train_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df, 
                    ims_embeddings_df, 'Train', input_type, embedding_type, show_wandb_run_name
                    )
                pf.plot_pca(
                    test_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df,
                    ims_embeddings_df, 'Test', input_type, embedding_type, show_wandb_run_name
                    )
                break
        # if save_emb_pca_to_wandb:
        #     # true_embeddings, predicted_embeddings_flattened, chem_names = 
        #     preds_to_emb_pca_plot(predicted_embeddings, output_name_encodings, sorted_chem_names, embedding_df)

        # at last epoch print model architecture details (this will also show up in wandb log)
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Dataset: ', wandb_kwargs['dataset'])
        print('Target Embeddings: ', wandb_kwargs['target_embedding'])
        print('-------------------------------------------')
        print('-------------------------------------------')
        print(model)
        print('-------------------------------------------')
        print('-------------------------------------------')

        # wandb.finish()

    print('Hyperparameters for best model: ')
    for key in best_hyperparams:
        print('   ', key, ' : ', best_hyperparams[key])
    
    return best_hyperparams

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def create_dataset_tensors(spectra_dataset, embedding_df, device, start_idx=None, stop_idx=None):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals, with 'Embedding Floats' 
        column corresponding to ChemNet embeddings.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure 
        (specifically without an 'Unnamed: 0' column). Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    spectra = spectra_dataset.iloc[:,start_idx:stop_idx]

    chem_encodings = spectra_dataset.iloc[:,-8:]

    # create tensors of spectra, true embeddings, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['Label'])
    embeddings_tensor = torch.Tensor([embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    chem_encodings_tensor = torch.Tensor(chem_encodings.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def create_dataset_tensors_for_generator(carl_dataset, embedding_preds, device=None, multiple_carls_per_spec=False, start_idx=2, stop_idx=-9):
    """
    Create tensors from the provided CARL dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing CARL data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing encoder predicted embeddings.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the CARLS.
    """
    # drop first col ('index') and last 9 cols ('Label', OneHot encodings) to get just CARLS and predicted embeddings
    # if carl:
    #     carls = carl_dataset.iloc[:,1:-9]
    # else:
    carls = carl_dataset.iloc[:,start_idx:stop_idx]

    # embeddings df doesn't have 'Label' col, so dropping last 8 cols instead of last 9
    embedding_preds = embedding_preds.iloc[:,1:-8]

    if carl_dataset.columns[-1] == 'Label':
        chem_encodings = carl_dataset.iloc[:,-9:-1]
    else:
        chem_encodings = carl_dataset.iloc[:,-8:]
    # del carl_dataset

    if device is not None:
        # Ensure all columns are numeric before converting to torch.Tensor
        # If not, attempt to convert or drop non-numeric columns
        if not all([np.issubdtype(dtype, np.number) for dtype in embedding_preds.dtypes]):
            embedding_preds = embedding_preds.apply(pd.to_numeric, errors='coerce')
            embedding_preds = embedding_preds.fillna(0)
        embeddings_preds = torch.Tensor(embedding_preds.values).to(device)
        carls = torch.Tensor(carls.values).to(device)
        chem_encodings = torch.Tensor(chem_encodings.values).to(device)
    else:
        embeddings_preds = torch.Tensor(embedding_preds.values)
        carls = torch.Tensor(carls.values)
        chem_encodings = torch.Tensor(chem_encodings.values)

    if multiple_carls_per_spec:
        # torch.Tensor changes the vals after decimal but I need those to stay the same so using torch.tensor instead
        carl_indices = torch.tensor(carl_dataset['index']).to(device)
    else:
        carl_indices = torch.Tensor(carl_dataset['index'].values).to(device)

    return embeddings_preds, carls, chem_encodings, carl_indices

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def create_dataset_tensors_with_dask(spectra_file, embedding_df, device, carl=False):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame using Dask.

    Parameters:
    ----------
    spectra_file : str
        Path to the CSV file containing spectral data and chemical labels.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor)
        - spectra_tensor (torch.Tensor)
        - chem_encodings_tensor (torch.Tensor)
        - spectra_indices_tensor (torch.Tensor)
    """
    # Load the dataset as a Dask DataFrame
    spectra_dd = dd.read_csv(spectra_file)

    # Compute the necessary tensors
    if carl:
        spectra = spectra_dd.iloc[:, 1:-9]
    else:
        spectra = spectra_dd.iloc[:, 2:-9]

    chem_labels = spectra_dd['Label'].compute().tolist()
    embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]

    # Create tensors directly from Dask DataFrame
    spectra_tensor = torch.tensor(spectra.compute().values, dtype=torch.float32).to(device)
    chem_encodings = spectra_dd.iloc[:, -8:].compute()
    chem_encodings_tensor = torch.tensor(chem_encodings.values, dtype=torch.float32).to(device)
    spectra_indices_tensor = torch.tensor(spectra_dd['index'].compute().values, dtype=torch.float32).to(device)

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def create_dataset_tensors_from_chunks(spectra_dataset, embedding_df, device, chunk_size=None, carl=False):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals, with 'Embedding Floats' 
        column corresponding to chemical names.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure 
        (specifically without an 'Unnamed: 0' column). Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    embeddings_list = []
    spectra_list = []
    chem_encodings_list = []
    indices_list = []

    # Process the dataset in chunks
    for chunk in pd.read_csv(spectra_dataset, chunksize=chunk_size):
        if carl:
            spectra = chunk.iloc[:, 1:-9]
            chem_labels = list(chunk['Label'])
            embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]
        else:
            spectra = chunk.iloc[:, 2:-9]
            chem_labels = list(chunk['Label'])
            embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]

        # Convert to tensors
        embeddings_tensor = torch.Tensor(embeddings).to(device)
        spectra_tensor = torch.Tensor(spectra.values).to(device)
        chem_encodings = chunk.iloc[:, -8:]
        chem_encodings_tensor = torch.Tensor(chem_encodings.values).to(device)
        spectra_indices_tensor = torch.Tensor(chunk['index'].to_numpy()).to(device)

        # Append to lists
        embeddings_list.append(embeddings_tensor)
        spectra_list.append(spectra_tensor)
        chem_encodings_list.append(chem_encodings_tensor)
        indices_list.append(spectra_indices_tensor)

    # Concatenate all tensors
    embeddings_tensor = torch.cat(embeddings_list).to(device)
    spectra_tensor = torch.cat(spectra_list).to(device)
    chem_encodings_tensor = torch.cat(chem_encodings_list).to(device)
    spectra_indices_tensor = torch.cat(indices_list).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
        nn.Linear(512,652),
        nn.LeakyReLU(inplace=True),
        nn.Linear(652,780),
        nn.LeakyReLU(inplace=True),
        nn.Linear(780, 908),
        nn.LeakyReLU(inplace=True),
        nn.Linear(908, 1036),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1036, 1164),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1164, 1292),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1292, 1420),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1420, 1548),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1548, 1676),
        )

    def forward(self, x):
        x = self.generator(x)
        return x
    
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
class ConditionGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
        nn.Linear(514,652),
        nn.LeakyReLU(inplace=True),
        nn.Linear(652,780),
        nn.LeakyReLU(inplace=True),
        nn.Linear(780, 908),
        nn.LeakyReLU(inplace=True),
        nn.Linear(908, 1036),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1036, 1164),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1164, 1292),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1292, 1420),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1420, 1548),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1548, 1676),
        )

    def forward(self, x):
        x = self.generator(x)
        return x

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class OneHottoIMSGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(8, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 780),
            nn.LeakyReLU(inplace=True),
            nn.Linear(780, 908),
            nn.LeakyReLU(inplace=True),
            nn.Linear(908, 1036),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1036, 1164),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1164, 1292),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1292, 1420),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1420, 1548),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1548, 1676),
        )

    def forward(self, x):
        x = self.generator(x)
        return x
   
def set_up_gpu():
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Current device ID:", device)
        print("PyTorch current device ID:", torch.cuda.current_device())
        print("PyTorch current device name:", torch.cuda.get_device_name(device))
    else:
        print("Using CPU")
        device = torch.device('cpu')
    return device


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def load_model(model_path, device, freeze_layers=6):
    print('Loading pretrained model...')
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    # Freeze first freeze_layers/2 Linear layers and LeakyReLU layers
    for layer in model.generator[:freeze_layers]: 
        for param in layer.parameters():
            param.requires_grad = False
    return model

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def train_generator(
        train_data, val_data, test_data, device, config, wandb_kwargs, 
        model_hyperparams, sorted_chem_names, generator_path, 
        save_plots_to_wandb = True, early_stop_threshold=10, 
        show_wandb_run_name=True, lr_scheduler = True, 
        num_plots = 1, patience=5, plot_overlap_pca=False, 
        model_type='Generator', pretrained_model_path=None,
        carl_or_spec = 'CARL'
        ):
    if save_plots_to_wandb:
        wandb.finish()
    # loss to compare for each model. Starting at infinity so it will be replaced by first model's first epoch loss 
    lowest_val_loss = np.inf

    keys = model_hyperparams.keys()
    values = model_hyperparams.values()

    # Generate all parameter combinations from model_config using itertools.product
    combinations = itertools.product(*values)

    # Iterate through each parameter combination and run model 
    for combo in combinations:
        combo = dict(zip(keys, combo))

        if early_stop_threshold > combo['epochs']:
            early_stop = combo['epochs']
        else:
            early_stop = early_stop_threshold
        # creating different var for model loss to use for early stopping
        lowest_val_model_loss = np.inf
        
        # load pretrained model if provided, otherwise create new model
        if pretrained_model_path is not None:
            model = load_model(pretrained_model_path, freeze_layers=combo['freeze_layers'])
        else:
            if model_type == 'Generator':
                model = Generator().to(device)
            elif model_type == 'OneHottoIMSGenerator':
                model = OneHottoIMSGenerator().to(device)
            elif model_type == 'ConditionGenerator':
                model = ConditionGenerator().to(device)

        epochs_without_validation_improvement = 0

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr = combo['learning_rate'])
        criterion = nn.MSELoss()

        final_lr = combo['learning_rate']

        if lr_scheduler:
            # Initialize the learning rate scheduler with patience of 5 epochs 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

        # wandb_kwargs = update_wandb_kwargs(wandb_kwargs, combo)

        # run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(1, combo['epochs']+1):
            if epochs_without_validation_improvement < early_stop:
                model.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if epoch == combo['epochs']:
                    average_loss, _, _ = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )
                    # wandb.log({'Learning Rate at Final Epoch':final_lr})
                    # save output plots to weights and biases
                    if save_plots_to_wandb:
                        pf.plot_and_save_generator_results(
                            train_data, combo['batch_size'], sorted_chem_names, 
                            model, device, criterion, num_plots, plot_overlap_pca=plot_overlap_pca, 
                            save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                            test_or_train='Train', carl_or_spec=carl_or_spec
                            )
                        pf.plot_and_save_generator_results(
                            test_data, combo['batch_size'], sorted_chem_names, 
                            model, device, criterion, num_plots, plot_overlap_pca=False, 
                            save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                            test_or_train='Test', carl_or_spec=carl_or_spec
                            )
            
                else:
                    average_loss = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )

                epoch_val_loss = 0  
                # evaluate model on validation data
                model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch, val_name_encodings, val_true_embeddings, _ in val_dataset:
                        val_batch = val_batch.to(device)
                        val_name_encodings = val_name_encodings.to(device)
                        val_true_embeddings = val_true_embeddings.to(device)

                        val_batch_predicted_embeddings = model(val_batch)

                        val_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)
                        # accumulate epoch validation loss
                        epoch_val_loss += val_loss.item()

                # divide by number of batches to calculate average loss
                val_average_loss = epoch_val_loss/len(val_dataset)

                if lr_scheduler:
                    scheduler.step(val_average_loss)  # Pass the validation loss to the scheduler
                    # get the new learning rate (to give to wandb)
                    final_lr = optimizer.param_groups[0]['lr']

                if val_average_loss < lowest_val_model_loss:
                    # check if val loss is improving for this model
                    epochs_without_validation_improvement = 0
                    lowest_val_model_loss = val_average_loss
                    # best_epoch = epoch  # Store the best epoch

                    if val_average_loss < lowest_val_loss:
                        # if current epoch of current model is best performing (of all epochs and models so far), save model state
                        # Save the model
                        # torch.save(model.state_dict(), generator_path)
                        torch.save(model, generator_path)
                        print(f'Saved best model at epoch {epoch}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                    else:
                        print(f'Model best validation loss at {epoch}')
                
                else:
                    epochs_without_validation_improvement += 1

                # log losses to wandb
                # wandb.log({f"{model_type} Training Loss": average_loss, f"{model_type} Validation Loss": val_average_loss})

                if (epoch) % 10 == 0 or epoch == 0:
                    print('Epoch[{}/{}]:'.format(epoch, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
    
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch}.')
                # wandb.log({'Early Stopping Epoch':epoch})
                # wandb.log({'Learning Rate at Final Epoch':final_lr})
                pf.plot_and_save_generator_results(
                    train_data, combo['batch_size'], sorted_chem_names, 
                    model, device, criterion, num_plots, plot_overlap_pca=plot_overlap_pca, 
                    save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                    test_or_train='Train', carl_or_spec=carl_or_spec
                    )
                pf.plot_and_save_generator_results(
                    test_data, combo['batch_size'], sorted_chem_names, 
                    model, device, criterion, num_plots, plot_overlap_pca=False, 
                    save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                    test_or_train='Test', carl_or_spec=carl_or_spec
                    )
                
                break

        # at last epoch print model architecture details (this will also show up in wandb log)
        print('-------------------------------------------')
        print('-------------------------------------------')
        if 'dataset' in wandb_kwargs.keys():
            print('Dataset: ', wandb_kwargs['dataset'])
        elif 'input' in wandb_kwargs.keys():
            print('Input data: ', wandb_kwargs['input'])
        if 'target_embedding' in wandb_kwargs.keys():
            print('Target Embeddings: ', wandb_kwargs['target_embedding'])
        elif 'target' in wandb_kwargs.keys():
            print('Target: ', wandb_kwargs['target'])
        print('-------------------------------------------')
        print('-------------------------------------------')
        print(model)
        print('-------------------------------------------')
        print('-------------------------------------------')

        # wandb.finish()

    print('Hyperparameters for best model: ')
    for key in best_hyperparams:
        print('   ', key, ' : ', best_hyperparams[key])
    
    return best_hyperparams

# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    

def predict_carls(dataset, model, device, criterion):
    total_loss = 0

    model.eval() # Set model to evaluation mode
    predicted_carls = []
    output_name_encodings = []
    input_carl_indices = []

    with torch.no_grad():
        for batch, name_encodings, true_carls, carl_indices in dataset:
            batch = batch.to(device)
            true_carls = true_carls.to(device)

            batch_predicted_carls = model(batch)
            predicted_carls.append(batch_predicted_carls)
            output_name_encodings.append(name_encodings)
            input_carl_indices.append(carl_indices)

            # print(batch_predicted_embeddings.shape, true_embeddings.shape)

            loss = criterion(batch_predicted_carls, true_carls)
            # accumulate loss
            total_loss += loss.item()

    # divide by number of batches to calculate average loss
    average_loss = total_loss/len(dataset)
    return predicted_carls, output_name_encodings, average_loss, input_carl_indices

# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------  

def generate_average_bkg_spectrum(bkg_df, n, num_avg_spectra=1, random_state=42):
    avg_spectra = []
    for _ in range(num_avg_spectra):
        bkg_sample = bkg_df.sample(n=n, random_state=random_state).iloc[:,1:-1]
        bkg_df = bkg_df.drop(bkg_sample.index)
        bkg_sample.reset_index(inplace=True)
        bkg_sample.drop(columns=['index'], inplace=True)

        avg_spectrum = bkg_sample.mean()
        avg_spectra.append(avg_spectrum)

    return avg_spectra