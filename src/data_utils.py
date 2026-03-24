from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report

## Split it into training and test sets for ML
def split_my_data(df, target_col):
    """
    Standard 80/20 split for training and testing.
    """
    # X = everything except the answer
    X = df.drop(columns=[target_col])
    
    # y = just the answer
    y = df[target_col]
    
    # The actual split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

## Scale my columns 
def scale_numeric_columns(X_train, X_test, columns_to_scale):
    """
    Scale specified numeric columns using StandardScaler.
    Returns scaled X_train, X_test and the fitted scaler.

    Parameters:
    - X_train, X_test: pandas DataFrames
    - columns_to_scale: list of column names to scale

    Returns:
    - X_train_scaled, X_test_scaled: scaled DataFrames
    - scaler: fitted StandardScaler object
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":

    # Dummy dataset
    data = {
        'SpecialDayProximity': [0.0, 0.2, 0.5, 0.1, 0.3],
        'ExitRate': [0.2, 0.1, 0.3, 0.25, 0.15],
        'PageValue': [0, 10, 20, 5, 15],
        'BounceRate': [0.2, 0.0, 0.3, 0.1, 0.05],
        'ProductPageTime': [0, 50, 100, 30, 70],
        'PurchaseCompleted': [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    print("Original Data:")
    print(df)

    # ---- Test split function ----
    X_train, X_test, y_train, y_test = split_my_data(df, 'PurchaseCompleted')

    print("\nX_train:")
    print(X_train)
    print("\nX_test:")
    print(X_test)

    # ---- Test scaling function ----
    numeric_cols = [
        'SpecialDayProximity', 
        'ExitRate', 
        'PageValue', 
        'BounceRate', 
        'ProductPageTime'
    ]

    X_train_scaled, X_test_scaled, scaler = scale_numeric_columns(X_train, X_test, numeric_cols)

    print("\nScaled X_train:")
    print(X_train_scaled)

    print("\nScaled X_test:")
    print(X_test_scaled)