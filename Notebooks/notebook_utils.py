from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

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