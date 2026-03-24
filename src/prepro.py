import pandas as pd 
import logging

# Set up logging to track what's happening
#logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

## convert negative values from productpagetime and bouncerate to nan first 
def convert_tonan(df):
    """ Convert our negative values from Bouncerate and productpagetime to nan """
    
    df.loc[df['BounceRate'] < 0, 'BounceRate'] = pd.NA
    df.loc[df['ProductPageTime'] < 0, 'ProductPageTime'] = pd.NA
    
    logging.info(f"BounceRate negatives: {df['BounceRate'].isna().sum()}")
    logging.info(f"ProductPageTime negatives: {df['ProductPageTime'].isna().sum()}")
        
    return df
    
## impute the nan values with median. 
def impute_median(df, col):
    
    for col in col:
        df[col] = df[col].fillna(df[col].median())

    return df 
## group geograhic region together 

def group_GR(df, col_name):
    
    df_copy = df.copy()
    if col_name == 'GeographicRegion':
        df_copy[col_name] = df_copy[col_name].apply(lambda x: 'Unknown' if x < 1 else x)
    return df_copy

## Collapse all labels into New, Returning, or Unregistered.
def map_customer_types(df):
    
    mapping = {
        "new_visitor": "new_visitor",
        "returning_visitor": "returning_visitor"
    }

    df["CustomerType"] = (
        df["CustomerType"]
        .astype(str)
        .str.lower()
        .map(mapping)
        .fillna("unregistered")
    )

    return df

## finding out categories in trafficsource <1% and relabel them as other
def group_low_frequency(df, col_name, threshold=0.01):
    """
    Group categories in a column that occur in less than `threshold` fraction of rows into 'Other'.
    """
    df_copy = df.copy()
    total_rows = len(df_copy)
    value_counts = df_copy[col_name].value_counts()
    low_freq_categories = value_counts[value_counts / total_rows < threshold].index
    df_copy[col_name] = df_copy[col_name].apply(lambda x: 'Other' if x in low_freq_categories else x)
    return df_copy

## creating a new category called unknown, allowing us to understand if missingness helps with anything
def add_unknown_category(df, col_name):
    """
    Replace NaN values in a categorical column with 'Unknown'.
    """
    df_copy = df.copy()
    df_copy[col_name] = df_copy[col_name].fillna('Unknown')
    return df_copy



## we will test our functions here to see if it works. 
if __name__ == "__main__":
    
    # Dummy test data
    data = {
        'BounceRate': [50, -10, None],
        'ProductPageTime': [5, None, 10],
        'ExitRate': [2, 10, 3]
    }
    df = pd.DataFrame(data)
    print("Before:")
    print(df)

    
    ## test the convert_tonan function
    df1 = convert_tonan(df.copy())
    print("\nAfter convert_tonan:")
    print(df1)

    
    ## test the impute_median function
    num_cols = ['BounceRate', 'ProductPageTime', 'ExitRate']
    df2 = impute_median(df1.copy(), num_cols)
    print("\nAfter impute_median:")
    print(df2)

    
    ## test GeographicRegion code
    # Test data
    GR = {'GeographicRegion': [1, 2, -1, 3, -2]}
    df_GR = pd.DataFrame(GR)

    df_GR_test = group_GR(df_GR, 'GeographicRegion')
    print("After grouping:")
    print(df_GR_test)

    
    ##test mapcustomer function
    data_customer = {
        "CustomerType": ["New_Visitor", "returning_Visitor", "Guest", 'Unknown']
    }
    df_customer = pd.DataFrame(data_customer)
    df_customer_test = map_customer_types(df_customer.copy())
    print(df_customer_test)