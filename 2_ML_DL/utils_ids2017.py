from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import RandomOverSampler
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

def load_dataset(file_path):
        df = pd.read_csv(file_path)
        convert_dict = {'label': 'category'}
        df = df.astype(convert_dict)
        df.info()
        return df
    
def replace_invalid(df):
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Identify columns with NaN, infinite, or negative values
        nan_columns = df[numeric_columns].columns[df[numeric_columns].isna().any()]
        inf_columns = df[numeric_columns].columns[np.isinf(df[numeric_columns]).any()]

        # Drop rows with NaN values (low percentage of NaN values)
        df = df.dropna(subset=nan_columns)

        # Drop rows with infinite values (assuming low percentage)
        for col in inf_columns:
            df = df[np.isfinite(df[col])]
        
        return df
    
def load_ids2017():
    file_path = r"..\CIC-IDS-2017\CSVs\GeneratedLabelledFlows\TrafficLabelling\processed\ids2017_processed.csv"
    df = load_dataset(file_path)
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
    X = X.drop(columns=['destination_port'])
    X = correlation_feature_selection(X)
    # Determine the selected features using the oversampled subset
    selected_features = information_gain_feature_selection(X, Y)
    # Apply the selected features to the main dataset
    X = X[selected_features]
    X.info()
    return X
    