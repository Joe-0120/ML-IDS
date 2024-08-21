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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import re
import gc
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import mutual_info_classif

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

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform the labels to integers
    combined_df['label_code'] = label_encoder.fit_transform(combined_df['label'])

    # Double the number of rows with 'label' = "SQL Injection" by copying them
    sql_injection_rows = combined_df[combined_df['label'] == 'SQL Injection']
    combined_df = pd.concat([combined_df, sql_injection_rows])
    
    return combined_df

def replace_invalid(df):
    df = df.drop(columns=['timestamp'])
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    # Identify columns with NaN, infinite, or negative values
    nan_columns = df[numeric_columns].columns[df[numeric_columns].isna().any()]
    inf_columns = df[numeric_columns].columns[np.isinf(df[numeric_columns]).any()]
    # Drop rows with NaN values (low percentage of NaN values)
    df = df.dropna(axis=0)
    # Drop rows with infinite values (assuming low percentage)
    for col in inf_columns:
        df = df[np.isfinite(df[col])]
    return df
    
def load_ids2018():
    file_path = r"..\CIC-IDS-2018\Processed Traffic Data for ML Algorithms"
    df = load_sample_dataset_2018(file_path)
    df = replace_invalid(df)
    return df

def oversample_minority_classes(X, Y, sample_size=1000):
    y = Y["label_code"]
    # Create a subset of the oversampled data
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_sample, y_sample)
    return X_resampled, y_resampled

def information_gain_feature_selection(X, Y, sample_size=1000):
    # Create an oversampled subset of the data
    X_sample, y_sample = oversample_minority_classes(X, Y, sample_size)
    # Create is_attack column based on label_code
    y_sample = (y_sample != 0).astype(int)
    # Perform feature selection on the oversampled subset
    info_gain = mutual_info_classif(X_sample, y_sample)
    info_gain_df = pd.DataFrame({'Feature': X.columns, 'Information Gain': info_gain})
    info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
    print(info_gain_df)
    selected_features = info_gain_df[info_gain_df['Information Gain'] > 0.1]['Feature'].tolist()
    return selected_features
   
def correlation_feature_selection(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def feature_selection(X, Y):
    stats = X.describe()
    std = stats.loc["std"]
    features_no_var = std[std == 0.0].index
    # Exclude non-numeric columns (e.g., categorical columns) from the features with zero variance
    features_no_var_numeric = [col for col in features_no_var if col in X.select_dtypes(include=[np.number]).columns]
    X = X.drop(columns=features_no_var_numeric)
    X = X.drop(columns=['dst_port'])
    X = correlation_feature_selection(X)
    # Determine the selected features using the oversampled subset
    selected_features = information_gain_feature_selection(X, Y)
    # Apply the selected features to the main dataset
    X = X[selected_features]
    X.info()
    return X