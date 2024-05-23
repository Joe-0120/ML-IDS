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

attack_labels = {
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

def upsample_dataset(X, Y, min_samples, attack_labels):
    Y = Y.drop(columns=['label'])
    # Combine X and Y to ensure we can apply SMOTE
    combined = pd.concat([X, Y], axis=1)
    # Get the counts of each class in label_code
    counts = Y['label_code'].value_counts()
    # Create a dictionary with the sampling strategy
    samples_number = {i: max(counts[i], min_samples) for i in np.unique(Y['label_code'])}
    # Convert to numpy arrays for SMOTE
    combined_array = combined.values
    y_array = Y['label_code'].values  # This will be used as the target for SMOTE
    # Apply SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=samples_number)
    resampled_array, y_resampled = smote.fit_resample(combined_array, y_array)
    
    # Split resampled array back into X and Y
    X_resampled = resampled_array[:, :-Y.shape[1]]  # X columns are all but the last Y.shape[1] columns
    Y_resampled = resampled_array[:, -Y.shape[1]:]  # Y columns are the last Y.shape[1] columns
    
    # Convert back to DataFrame if needed
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    Y_resampled_df = pd.DataFrame(Y_resampled, columns=Y.columns)
    # Add the 'label' column based on attack_labels dictionary
    Y_resampled_df['label'] = Y_resampled_df['label_code'].map(attack_labels)
    Y_resampled_df['label'] = Y_resampled_df['label'].astype('category')
    return X_resampled_df, Y_resampled_df

def replace_invalid(df):
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Identify columns with NaN, infinity, or negative values
    invalid_columns = df[numeric_columns].columns[df[numeric_columns].isna().any() |
                                                  np.isinf(df[numeric_columns]).any() |
                                                  (df[numeric_columns] < 0).any()]

    # print("Columns with NaN, infinity, or negative values:", invalid_columns.tolist())
    
    # Replace invalid values with NaN and fill with column mean
    df[invalid_columns] = df[invalid_columns].replace([np.inf, -np.inf, -1], np.nan)
    df[invalid_columns] = df[invalid_columns].fillna(df[invalid_columns].mean())
    
    return df

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    convert_dict = {'label': 'category'}
    df = df.astype(convert_dict)
    replace_invalid(df)
    # df.info()
    return df

def load_processed_dataset_2017(file_path):
    df = load_dataset(file_path)
    X = df.iloc[:, 0:79]
    Y = df.iloc[:, 79:]
    stats = X.describe()
    std = stats.loc["std"]
    features_no_var = std[std == 0.0].index
    # Exclude non-numeric columns (e.g., categorical columns) from the features with zero variance
    features_no_var_numeric = [col for col in features_no_var if col in X.select_dtypes(include=[np.number]).columns]
    X = X.drop(columns=features_no_var)
    X = X.drop(columns=['destination_port'])
    threshold = 0.9
    corr_matrix = X.corr().abs()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    to_keep = [
    'Destination Port', 'Fwd Packet Length Std', 'Min Packet Length', 
    'Packet Length Variance', 'PSH Flag Count', 'Active Max'
    ]
    to_drop = [column for column in to_drop if column not in to_keep]
    X = X.drop(columns=to_drop)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, stratify=Y.label_code)
    X_eval, X_test, Y_eval, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp.label_code)
    X_train, Y_train = upsample_dataset(X, Y, 100000, attack_labels)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return X_train, Y_train, X_eval, Y_eval, X_test, Y_test, scaler

import matplotlib.pyplot as plt

def plot_confusion_matrix(model_name, Y_true, Y_pred, labels=["Benign", "Attack"]):
    matrix = confusion_matrix(Y_true.is_attack, Y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def metrics_report(dataset_type, y_true, y_predict, print_avg=True):
    print(f"Classification Report ({dataset_type}):")
    print(classification_report(y_true, y_predict, digits=4))
    if print_avg:
        print(f"Avg Precision Score: {average_precision_score(y_true, y_predict, average='weighted')}")
    print("Accuracy:",accuracy_score(y_true, y_predict))
    res = classification_report(y_true, y_predict, digits=4, output_dict = True)
    res["accuracy"] = accuracy_score(y_true, y_predict)
    return res

def extract_and_plot_metrics(metrics_dict):
    # Initialize dictionaries to store the metrics for plotting
    precision_dict = {'0': [], '1': [], 'model': []}
    recall_dict = {'0': [], '1': [], 'model': []}
    f1_score_dict = {'0': [], '1': [], 'model': []}
    accuracy_list = []

    # Iterate over the models in the metrics dictionary
    for model_name, metrics in metrics_dict.items():
        precision_dict['0'].append(metrics['0']['precision'])
        precision_dict['1'].append(metrics['1']['precision'])
        recall_dict['0'].append(metrics['0']['recall'])
        recall_dict['1'].append(metrics['1']['recall'])
        f1_score_dict['0'].append(metrics['0']['f1-score'])
        f1_score_dict['1'].append(metrics['1']['f1-score'])
        accuracy_list.append(metrics['accuracy'])
        precision_dict['model'].append(model_name)
        recall_dict['model'].append(model_name)
        f1_score_dict['model'].append(model_name)
        
    # Plotting the metrics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot precision
    axs[0, 0].plot(precision_dict['model'], precision_dict['0'], label='Class 0', marker='o')
    axs[0, 0].plot(precision_dict['model'], precision_dict['1'], label='Class 1', marker='o')
    axs[0, 0].set_title('Precision')
    axs[0, 0].legend()
    
    # Plot recall
    axs[0, 1].plot(recall_dict['model'], recall_dict['0'], label='Class 0', marker='o')
    axs[0, 1].plot(recall_dict['model'], recall_dict['1'], label='Class 1', marker='o')
    axs[0, 1].set_title('Recall')
    axs[0, 1].legend()
    
    # Plot f1-score
    axs[1, 0].plot(f1_score_dict['model'], f1_score_dict['0'], label='Class 0', marker='o')
    axs[1, 0].plot(f1_score_dict['model'], f1_score_dict['1'], label='Class 1', marker='o')
    axs[1, 0].set_title('F1-Score')
    axs[1, 0].legend()
    
    # Plot accuracy
    print(accuracy_list)
    print(precision_dict['model'])
    axs[1, 1].plot(precision_dict['model'], accuracy_list, label='Accuracy', marker='o')
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()