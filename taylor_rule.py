"""
Central Bank Data Analysis and Interest Rate Prediction Script

This script connects to a MySQL database to retrieve economic data for a specified central bank (either FED or BCE),
performs data preprocessing and analysis, and uses machine learning to predict interest rates. The results are then
plotted for visualization.

Modules and Libraries Used:
- `pandas`: For data manipulation and analysis.
- `sklearn`: For machine learning models and training.
- `matplotlib`: For plotting graphs.
- `sqlalchemy`: For creating engine connections to the database.
- `mysql-connector-python`: For connecting and executing queries in MySQL database.

Functions:
- `create_server_connection(host_name, user_name, user_password, db_name)`: Establishes and returns a connection to the MySQL server.
- `get_data_for_central_bank(connection, bank)`: Retrieves and processes data from the database for the specified central bank.
- `train_and_predict(df, INTEREST_RATE_POT, output_gap, inflation_gap)`: Trains a linear regression model on the data and makes predictions for the interest rate.
- `plot_results(date, actual_rate, predicted_rate)`: Plots the actual vs. predicted interest rates for visualization.
- `main()`: Main function that coordinates the data retrieval, processing, training, prediction, and plotting.

Usage:
1. Ensure that the MySQL database and required tables are properly set up and populated with the relevant data.
2. Modify the database connection parameters (`host_name`, `user_name`, `user_password`, `db_name`) as needed.
3. Run the script. When prompted, input 'FED' or 'BCE' to select the central bank for which you want to perform the analysis.

Dependencies:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `sqlalchemy`
- `mysql-connector-python`

Make sure these libraries are installed in your Python environment. You can install them using pip:
    `pip install pandas scikit-learn matplotlib sqlalchemy mysql-connector-python`
"""

from login_mysql import create_server_connection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def get_data_for_central_bank(connection, bank):
    """
    Retrieve and preprocess economic data for a specified central bank (either FED or BCE) from a MySQL database.

    Parameters:
    - connection (mysql.connector.connection_cext.CMySQLConnection): A connection object to the MySQL database.
    - bank (str): The central bank for which to retrieve data. Acceptable values are 'FED' or 'BCE'.

    Returns:
    - df (DataFrame): A DataFrame containing the processed data with columns 'funds_rate', 'output_gap', and 'inflation_gap'.
    - INTEREST_RATE_POT (Series): A Series containing potential interest rate values.
    - output_gap (Series): A Series containing the calculated output gap.
    - inflation_gap (Series): A Series containing the calculated inflation gap.
    - INTEREST_RATE['date'] (Series): A Series containing the dates corresponding to the interest rate data points.
    """
    # Define the data tickers, frequencies, and start dates based on the selected central bank
    if bank == 'FED':
        GDP_TICKER = 'GDPC1'
        GDP_FREQUENCY = 'Q'
        GDPPOT_TICKER = 'GDPPOT'
        GDPPOT_FREQUENCY = 'Q'
        CPI_TICKER = 'CPILFESL'
        CPI_FREQUENCY = 'Q'
        INTEREST_RATE_TICKER = 'FEDFUNDS'
        INTEREST_RATE_FREQUENCY = 'Q'
        REAL_INTEREST_RATE_TICKER = 'REAINTRATREARAT1Y'
        REAL_INTEREST_RATE_FREQUENCY = 'Q'
        CPI_START_DATE = '1981-01-01'
        GDP_START_DATE = '1982-01-01'
    elif bank == 'BCE':
        GDP_TICKER = 'EA19GDPC'
        GDP_FREQUENCY = 'Q'
        GDPPOT_TICKER = 'EA19GDPPOT'
        GDPPOT_FREQUENCY = 'Q'
        CPI_TICKER = 'CP0000EZ17M086NEST'
        CPI_FREQUENCY = 'M'
        INTEREST_RATE_TICKER = 'MIR00EZM156N'
        INTEREST_RATE_FREQUENCY = 'M'
        REAL_INTEREST_RATE_TICKER = 'REAINTRATREARAT1Y'
        REAL_INTEREST_RATE_FREQUENCY = 'Q'
        CPI_START_DATE = '1999-01-01'
        GDP_START_DATE = '1995-01-01'
    elif bank == 'BOE':
        GDP_TICKER = 'LNSAGDPC'
        GDP_FREQUENCY = 'Q'
        GDPPOT_TICKER = 'LNSAGDPPOT'
        GDPPOT_FREQUENCY = 'Q'
        CPI_TICKER = 'CPIAUCNS'
        CPI_FREQUENCY = 'Q'
        INTEREST_RATE_TICKER = 'UKBRBASE'
        INTEREST_RATE_FREQUENCY = 'M'
        REAL_INTEREST_RATE_TICKER = 'REAINTRATREARAT1Y'
        REAL_INTEREST_RATE_FREQUENCY = 'Q'
        CPI_START_DATE = '1989-01-01'
        GDP_START_DATE = '1990-01-01'
    elif bank == 'BOJ':
        GDP_TICKER = 'JPNRGDP'
        GDP_FREQUENCY = 'Q'
        GDPPOT_TICKER = 'JPNRGDPPOT'
        GDPPOT_FREQUENCY = 'Q'
        CPI_TICKER = 'JPNRCPI'
        CPI_FREQUENCY = 'Q'
        INTEREST_RATE_TICKER = 'JPNIRATE'
        INTEREST_RATE_FREQUENCY = 'Q'
        REAL_INTEREST_RATE_TICKER = 'REAINTRATREARAT1Y'
        REAL_INTEREST_RATE_FREQUENCY = 'Q'
        CPI_START_DATE = '1981-01-01'
        GDP_START_DATE = '1982-01-01'
    else:
        raise ValueError("Invalid central bank. Please use 'FED', 'BCE', 'BOE' or 'BOJ'.")

    GDP = pd.read_sql(f"SELECT * FROM {GDP_TICKER}_{GDP_FREQUENCY} WHERE date >= '{GDP_START_DATE}'", connection)
    GDPPOT = pd.read_sql(f"SELECT * FROM {GDPPOT_TICKER}_{GDPPOT_FREQUENCY} WHERE date >= '{GDP_START_DATE}'", connection)
    CPI = pd.read_sql(f"SELECT * FROM {CPI_TICKER}_{CPI_FREQUENCY} WHERE date >= '{CPI_START_DATE}'", connection)
    
    CPI['value_shifted_4'] = CPI['value'].shift(4)
    CPI['inflation'] = ((CPI['value'] - CPI['value_shifted_4']) / CPI['value_shifted_4']) * 100

    merge = GDP.merge(GDPPOT, on='date', suffixes=('_GDP', '_GDPPOT'))
    merge = merge.merge(CPI, on='date', suffixes=('_GDP', '_CPI'))

    output_gap = (merge['value_GDP'] - merge['value_GDPPOT']) / merge['value_GDPPOT'] * 100
    inflation_gap = merge['inflation'] - 2
    
    INTEREST_RATE = pd.read_sql(f"SELECT * FROM {INTEREST_RATE_TICKER}_{INTEREST_RATE_FREQUENCY} WHERE date >= '{GDP_START_DATE}' AND date <= '2024-01-01'", connection)
    
    query = f"SELECT * FROM {REAL_INTEREST_RATE_TICKER}_{REAL_INTEREST_RATE_FREQUENCY} WHERE date >= '{GDP_START_DATE}' AND date <= '2024-01-01'"
    REAL_INTEREST_RATE = pd.read_sql(query, connection)
    REAL_INTEREST_RATE = pd.to_numeric(REAL_INTEREST_RATE[f'{REAL_INTEREST_RATE_TICKER}E'])

    #INTEREST_RATE_POT = REAL_INTEREST_RATE + 2
    INTEREST_RATE_POT = REAL_INTEREST_RATE + merge['inflation']

    data = {
        'funds_rate': INTEREST_RATE['value'],
        'output_gap': output_gap,
        'inflation_gap': inflation_gap,
    }

    df = pd.DataFrame(data)
    return df, INTEREST_RATE_POT, output_gap, inflation_gap, INTEREST_RATE['date']

def train_and_predict(df, INTEREST_RATE_POT, output_gap, inflation_gap):
    """
    Train a linear regression model to predict the interest rate based on the output gap and inflation gap.

    Parameters:
    - df (DataFrame): DataFrame containing the data with columns 'output_gap', 'inflation_gap', and 'funds_rate'.
    - INTEREST_RATE_POT (Series): Potential interest rate values.
    - output_gap (Series): Series containing the calculated output gap.
    - inflation_gap (Series): Series containing the calculated inflation gap.

    Returns:
    - model (LinearRegression): The trained linear regression model.
    - predict_interest_rate (Series): The predicted interest rates based on the model and provided gaps.
    - coefficients (ndarray): Coefficients of the trained linear regression model.
    - intercept (float): Intercept of the trained linear regression model.
    """
    X = df[['output_gap', 'inflation_gap']]
    y = df['funds_rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    coefficients = model.coef_
    intercept = model.intercept_

    predict_interest_rate = INTEREST_RATE_POT + (coefficients[0] * output_gap) + (coefficients[1] * inflation_gap)
    

    return model, predict_interest_rate, coefficients, intercept

def plot_results(date, actual_rate, predicted_rate):
    # Create a new figure with a specific size
    plt.figure(figsize=(12, 6))
    # Plot the actual and predicted interest rates
    plt.plot(date, actual_rate, label='Actual Interest Rate', color='blue')
    plt.plot(date, predicted_rate, label='Predicted Interest Rate', color='orange')

    # Set the x-axis label and the y-axis label
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    # Set the title of the plot and add a legend to the plot
    plt.title('Actual vs Predicted Interest Rates')
    plt.legend()
    # Enable the grid for better readability
    plt.grid(True)
    # Display the plot
    plt.show()

def main():
    """
    Main function to connect to the MySQL database, retrieve data for the specified central bank,
    train a linear regression model to predict interest rates, and plot the actual vs. predicted interest rates.

    Steps:
    1. Establish a connection to the MySQL database.
    2. Prompt the user to select a central bank ('FED' or 'BCE').
    3. Retrieve and preprocess the economic data for the selected central bank.
    4. Train a linear regression model to predict interest rates based on the output gap and inflation gap.
    5. Display the model coefficients and intercept.
    6. Calculate and display the correlation between actual and predicted interest rates.
    7. Plot the actual and predicted interest rates over time.
    """
    # Step 1: Establish a connection to the MySQL database
    connection = create_server_connection("127.0.0.1", "root", "Castagnole2024!", "sidan") 
    # Step 2: Prompt the user to select a central bank
    BC = input('Per quale Banca Centrale vuoi fare la previsione? (FED, BCE, BOE or BOJ): ')  

    # Step 3: Retrieve and preprocess the economic data for the selected central bank
    df, INTEREST_RATE_POT, output_gap, inflation_gap, date = get_data_for_central_bank(connection, BC) 
    # Step 4: Train a linear regression model to predict interest rates
    model, predict_interest_rate, coefficients, intercept = train_and_predict(df, INTEREST_RATE_POT, output_gap, inflation_gap) 

    # Step 5: Display the model coefficients and intercept
    print(f'Coefficients: {coefficients}')
    print(f'Intercept: {intercept}')
    # Prepare data for plotting
    df_graph = pd.DataFrame({
        'date': date,
        'Interest_Rate': df['funds_rate'],
        'Forecast_Interest_Rate': predict_interest_rate,
    })
    # Step 6: Calculate and display the correlation between actual and predicted interest rates
    correlation = df_graph['Interest_Rate'].corr(df_graph['Forecast_Interest_Rate'])
    print(f"The correlation between {BC} Interest Rate and Forecast Interest Rate is: {correlation:.3f}")
    # Step 7: Plot the actual and predicted interest rates
    plot_results(df_graph['date'], df_graph['Interest_Rate'], df_graph['Forecast_Interest_Rate'])

if __name__ == "__main__":
    main()