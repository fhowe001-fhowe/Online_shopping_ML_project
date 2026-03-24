import sqlite3
import pandas as pd
import logging 

#db_path would be the file name like "online_shopping.db" and the table_name would just be "online_shopping"
#we will only use ingest to help us convert our db files to pandas df, and reusing this to help us do it for the same file with new data. 

def get_data_from_db(db_path, table_name):
    # Connect to your database
    try:
        conn = sqlite3.connect(db_path)
        # Pull the data
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        logging.info(f"Successfully loaded {len(df)} rows from {table_name}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise
        
if __name__ == "__main__":
    # Test loading the actual DB
    db_path = "data/online_shopping.db"
    table_name = "online_shopping"
    df = get_data_from_db(db_path, table_name)
    print(df.head(3))