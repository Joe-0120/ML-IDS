from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import os
import re
import gc
import seaborn as sns

attack_labels_2017 = {
    0: 'BENIGN',
    7: 'FTP-Patator',
    11: 'SSH-Patator',
    6: 'DoS slowloris',
    5: 'DoS Slowhttptest',
    4: 'DoS Hulk',
    3: 'DoS GoldenEye',
    8: 'Heartbleed',
    12: 'Web Attack - Brute Force',
    14: 'Web Attack - XSS',
    13: 'Web Attack - Sql Injection',
    9: 'Infiltration',
    1: 'Bot',
    10: 'PortScan',
    2: 'DDoS'
}

def load_sample_dataset_2018(file_path):
    # Define the regular expression to match spaces and special characters
    column_name_regex = re.compile(r'[^\w\s]')
    
    # Function to trim column names
    def trim_column_names(df):
        df.columns = [column_name_regex.sub('_', c.lower()) for c in df.columns]
        return df
        
    # Initialize an empty list to hold the sampled DataFrames
    df_list = []
    # Fraction to sample
    sampling_fraction = 0.1
    # Iterate over all CSV files in the folder
    for i, file_name in enumerate(os.listdir(file_path)):
        if file_name.endswith(".csv"):
            file_full_path = os.path.join(file_path, file_name)
            # Read the CSV file in chunks
            for chunk in pd.read_csv(file_full_path, chunksize=100000, low_memory=False):
                # Sample the chunk
                sampled_chunk = chunk.sample(frac=sampling_fraction, random_state=1)
                df_list.append(sampled_chunk)
                # Delete chunk to free memory
                del chunk
            # Print progress
            print(f"Processed {i+1}/{len(os.listdir(file_path))} files.")
    # Concatenate the sampled DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)
    # Apply the function to the column names
    combined_df = trim_column_names(combined_df)
    # Delete the list of DataFrames to free memory
    del df_list
    gc.collect()
    
    def replace_spaces_in_column_names(df):
        df.columns = [c.replace(' ', '_').lower() for c in df.columns]
        return df
    combined_df = replace_spaces_in_column_names(combined_df)

    # Remove irrelevant features
    combined_df = combined_df.iloc[:, 0:80]
    
    print("Creating is_attack column...")
    # Selecting the necessary columns and creating is_attack
    combined_df['is_attack'] = combined_df.label.apply(lambda x: 0 if x == "Benign" else 1)

    convert_dict = {'label': 'category'}
    combined_df = combined_df.astype(convert_dict)

    # Convert all object columns to numeric values
    combined_df = combined_df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' else x)
    # Verify the conversion
    print(combined_df.info())
    
    # Remove rows where the 'label' column has the value 'label'
    combined_df = combined_df[combined_df['label'] != 'label']
    return combined_df
