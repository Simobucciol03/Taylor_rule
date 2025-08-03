"""
FRED Data Import and MySQL Integration Script

This Python script fetches economic time series data from the FRED API and imports it into a MySQL database. 
It uses `requests` for API calls, `pandas` for data manipulation, and `mysql-connector-python` for database operations.

Functions:

- fetch_fred_data(series_id, api_key):
    Retrieves data from the FRED API for a specified series ID and frequency.
    - Parameters:
        - series_id (str): The ID of the economic data series to fetch.
        - api_key (str): Your API key for accessing FRED data.
    - Returns:
        - List of observations from the FRED API response.

- transform_data(observations):
    Converts raw data from the API into a pandas DataFrame and processes it.
    - Parameters:
        - observations (list): Raw data fetched from the FRED API.
    - Returns:
        - A pandas DataFrame with `date` and `value` columns; `value` is converted to a numeric type.

- load_data_to_mysql(df, connection):
    Inserts the processed DataFrame into a MySQL database table, creating the table if it doesn’t exist.
    - Parameters:
        - df (DataFrame): The DataFrame with processed data to be inserted.
        - connection (MySQLConnection): The MySQL database connection object.
    - Behavior:
        - Creates a table named `{FRED_SERIES_ID}_{FRED_SERIES_FREQUENCY}` if it doesn’t already exist.
        - Inserts the data into the specified table.

- main():
    Manages the overall workflow of the script:
    1. Establishes a connection to the MySQL database using `create_server_connection`.
    2. Fetches data from the FRED API using `fetch_fred_data`.
    3. Transforms the data into a DataFrame with `transform_data`.
    4. Loads the transformed data into the MySQL database using `load_data_to_mysql`.
    5. Queries and prints the data from the newly created MySQL table.
    6. Closes the database connection.

Usage:
1. Update the `FRED_API_KEY` with your FRED API key.
2. Run the script and input the series ID and frequency (e.g., daily, monthly, quarterly, or annual) when prompted.
3. Ensure that the MySQL connection parameters in `create_server_connection` match your database credentials.

Dependencies:
- requests: For making HTTP requests to the FRED API. Install with:
    pip install requests
- pandas: For handling and processing data. Install with:
    pip install pandas
- mysql-connector-python: For interacting with MySQL. Install with:
    pip install mysql-connector-python
"""

import requests
import pandas as pd
from mysql.connector import Error
from login_mysql import create_server_connection

# Configure your FRED API Key
FRED_API_KEY = 'a50520100e4f74d3a78f283ebb2b1cfe'
FRED_SERIES_ID = input("Enter ID: ") # Sostituisci con l'ID della serie che desideri recuperare
FRED_SERIES_FREQUENCY = input("Enter the frequency (m = mounthly, q = quartely, a = annual): ")

def fetch_fred_data(series_id, api_key):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&frequency={FRED_SERIES_FREQUENCY}"
    response = requests.get(url)
    data = response.json()
    observations = data['observations']
    return observations


def transform_data(observations):
    df = pd.DataFrame(observations)
    df = df[['date', 'value']]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def load_data_to_mysql(df, connection):
    try:
        if connection.is_connected():
            cursor = connection.cursor()

            # Create the table if it does not exist
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {FRED_SERIES_ID}_{FRED_SERIES_FREQUENCY} (
                    date DATE,
                    value FLOAT
                )
            ''')

            # Enter the data into the database
            for _, row in df.iterrows():
                cursor.execute(f'''
                    INSERT INTO {FRED_SERIES_ID}_{FRED_SERIES_FREQUENCY} (date, value) VALUES (%s, %s)
                ''', tuple(row))
            
            connection.commit()
            cursor.close()
            print("Data loaded to MySQL successfully!")
        else:
            print("Database connection failed")
    except Error as e:
        print(f"Error: {e}")

def main():
    connection = create_server_connection("127.0.0.1", "root", "Castagnole2024!", "sidan")
    observations = fetch_fred_data(FRED_SERIES_ID, FRED_API_KEY)
    df = transform_data(observations)
    load_data_to_mysql(df, connection)
    data = pd.read_sql(f"SELECT * FROM {FRED_SERIES_ID}_{FRED_SERIES_FREQUENCY}", connection)
    print(data)
    if connection.is_connected():
        connection.close()

if __name__ == "__main__":
    main()