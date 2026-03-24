import pandas as pd
import numpy as np
import logging 
from sklearn.preprocessing import OneHotEncoder


## function for one-hot encoding
def apply_one_hot_encoding(df, columns_to_encode, encoder=None):
    df_encoded = df.copy()
    
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df_encoded[columns_to_encode])
    else:
        encoded_data = encoder.transform(df_encoded[columns_to_encode])
    
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=encoder.get_feature_names_out(columns_to_encode),
        index=df_encoded.index
    )
    
    df_encoded = df_encoded.drop(columns=columns_to_encode)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded, encoder


## function for log transform
def apply_log_transform(df, columns_to_transform):
    """
    Applies log1p transformation to specified columns to handle right-skewness. (productpagetime and the page value)
    Returns a new dataframe to avoid 'SettingWithCopyWarning'.
    """
    df_transformed = df.copy()
    for col in columns_to_transform:
        if col in df_transformed.columns:
            # We check if the column is already transformed (optional, but professional)
            # and use log1p to handle zeros gracefully.
            df_transformed[col] = np.log1p(df_transformed[col])
            logging.info(f"Applied log1p transformation to column: {col}")
        else:
            logging.warning(f"Column {col} not found in dataframe. Skipping.")
            
    return df_transformed

## test code functions

if __name__ == "__main__":

    ## testing our logtransform function
    test_data = {
        'ProductPageTime': [0, 5, 10],
        'PageValue': [100, 0, 50],
        'OtherCol': [1, 2, 3]
    }

    df_test_log = pd.DataFrame(test_data)

    # Specify which columns to transform
    df_log = apply_log_transform(df_test_log, ['ProductPageTime', 'PageValue'])
    print(df_log)


    ##testing our one-hot encode function
    df_test_OHE = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Female'],
        'Region': ['North', 'South', 'North'],
        'Value': [10, 20, 30]
    })

    df_encoded, encoder = apply_one_hot_encoding(df_test_OHE, ['Gender', 'Region'])
    print(df_encoded)