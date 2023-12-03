import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import matplotlib.dates as mdates
import time

# Database Connection
engine = create_engine('postgresql://stockeradminsimec:stock_admin_#146@localhost/stockerhubdb')

# Specify the schema and table name
schema_name = 'public'
table_name = 'company_price_companyprice'

instr_codes = pd.read_sql(f"SELECT \"Instr_Code\" FROM public.company_price_companyprice group by \"Instr_Code\"", engine)["Instr_Code"].tolist()

engine_sc = engine
selected_columns = ['mkt_info_date', 'closing_price']

# Function for the program
def holt_winters(data, alpha, beta, gamma, periods):
    n = len(data)
    level = data['closing_price'].iloc[0]  # Use column name 'closing_price'
    trend = np.mean(data['closing_price'].iloc[1:n] - data['closing_price'].iloc[0:n - 1])
    seasonality = np.array([data['closing_price'].iloc[i] - level - trend * i for i in range(n)])
    forecast = []

    for i in range(n, n + periods):
        level_old, trend_old = level, trend
        level = alpha * (data['closing_price'].iloc[i - n] - seasonality[i - n]) + (1 - alpha) * (level + trend)
        trend = beta * (level - level_old) + (1 - beta) * trend_old
        seasonality[i % n] = gamma * (data['closing_price'].iloc[i - n] - level) + (1 - gamma) * seasonality[i - n]
        forecast.append(level + i * trend + seasonality[i % n])

    return forecast

# Required Parameter


alpha = 0.2
beta = 0.1
gamma = 0.3
periods_to_forecast = 365

for target_instr_code in instr_codes:
    # Construct the SQL query to select data from the specified table and filter by name
    query = f"SELECT {', '.join(selected_columns)} FROM {table_name} WHERE \"Instr_Code\" = '{target_instr_code}'"
    # Execute the query and load the results into a DataFrame
    df = pd.read_sql(query, engine_sc)
    closing_prices = df['closing_price']

    forecast = holt_winters(df, alpha, beta, gamma, periods_to_forecast)
    print(instr_codes)
    print(f"Forecasted Closing Prices {target_instr_code}:")
    print(forecast)

    # Plotting the results
    df['mkt_info_date'] = pd.to_datetime(df['mkt_info_date'])
    df = df.sort_values('mkt_info_date')

    plt.figure(figsize=(10, 6))

    # Plot actual closing prices
    plt.plot(df['mkt_info_date'], df['closing_price'], label='Actual Closing Prices', marker='o')

    # Plot forecasted closing prices
    forecast_dates = pd.date_range(start=df['mkt_info_date'].iloc[-1], periods=periods_to_forecast + 1, freq='B')[1:]
    plt.plot(forecast_dates, forecast, label='Forecasted Closing Prices', linestyle='--', marker='o')

    # Customize the plot
    plt.title(f'Holt-Winters Forecasting for "{target_instr_code}"')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    time.sleep(3)
