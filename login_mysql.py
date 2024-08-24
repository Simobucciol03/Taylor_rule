"""
MySQL Database Connection Script

This script establishes a connection to a MySQL database using the `mysql-connector-python` library. 

Functions:
- `create_server_connection(host_name, user_name, user_password, db_name)`: 
    Establishes and returns a connection to a MySQL server using the specified parameters.
    - Parameters:
        - `host_name` (str): The hostname or IP address of the MySQL server.
        - `user_name` (str): The username for authenticating with the MySQL server.
        - `user_password` (str): The password for the specified username.
        - `db_name` (str): The name of the database to connect to.

    - Returns:
        - A connection object if the connection is successful.
        - `None` if an error occurs during the connection attempt.

Usage:
1. Modify the `database_name`, `host_name`, `user_name`, and `user_password` variables to match your MySQL server and database credentials.
2. Call the `create_server_connection` function with these parameters to establish a connection.

Dependencies:
- `mysql-connector-python`: Ensure this library is installed in your Python environment. You can install it using pip:
    `pip install mysql-connector-python`
"""

import mysql.connector
from mysql.connector import Error

def create_server_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            port=3306,
            user=user_name,
            password=user_password,
            database=db_name
        )
        print("MySQL Database connection successful, now you can create your magic ideas!")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

# Specifica il nome del tuo database qui
database_name = "sidan"

# Crea la connessione al database MySQL
connection = create_server_connection("127.0.0.1", "root", "Castagnole2024!", database_name)


