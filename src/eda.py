import pandas as pd
import numpy as np
import sqlite3
from db import get_connection

con = get_connection()
query = """
    SELECT 
        s.ticker,
        b.timestamp,
        b.open,
        b.high,
        b.low,
        b.close,
        b.volume,
        b.transactions
    FROM hourly_bars b
    JOIN symbols s ON s.id = b.symbol_id
    ORDER BY b.timestamp ASC;
"""
df = pd.read_sql_query(query, con)
con.close()

# Convert string to datetime in UTC
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Convert to Eastern Time
df['timestamp_et'] = df['timestamp'].dt.tz_convert('America/New_York')

# Extract hour and day of week
df['hour_et'] = df['timestamp_et'].dt.hour
df['day_name'] = df['timestamp_et'].dt.day_name()

print(df.head())
print(df.info())
print(df.describe())

# Average volume by hour of day (Eastern Time)
hourly_stats = df.groupby('hour_et')['volume'].mean()
print("\n--- AVERAGE VOLUME BY HOUR (ET) ---")
print(hourly_stats.apply(lambda x: f"{x:,.0f}"))

# Compare raw volume skew vs log volume skew
df['log_volume'] = np.log(df['volume'].clip(lower=1.0))

print("\n--- SKEWNESS (Normal distribution = ~0) ---")
print(f"Raw Volume Skew: {df['volume'].skew():.2f}")
print(f"Log Volume Skew: {df['log_volume'].skew():.2f}")

# Regular Market Hours (RTH) are 9:30 AM to 4:00 PM Eastern
rth = df[(df['hour_et'] >= 9) & (df['hour_et'] <= 16)]
eth = df[(df['hour_et'] < 9) | (df['hour_et'] > 16)]

print(f"\nRegular Hours Total Volume:  {rth['volume'].sum():,.0f} ({len(rth)} bars)")
print(f"Extended Hours Total Volume: {eth['volume'].sum():,.0f} ({len(eth)} bars)")

print(df['day_name'].value_counts())
