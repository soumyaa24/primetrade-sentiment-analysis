import pandas as pd, numpy as np

td_raw = pd.read_csv('data/trader_data.csv')
print('Timestamp IST unique count:', td_raw['Timestamp IST'].nunique())
print('Sample:', td_raw['Timestamp IST'].head(5).tolist())

# Parse with dayfirst=True (dd-mm-yyyy hh:mm)
td_raw['datetime_parsed'] = pd.to_datetime(td_raw['Timestamp IST'], dayfirst=True, errors='coerce')
print('Parse errors:', td_raw['datetime_parsed'].isna().sum())
print('Date range:', td_raw['datetime_parsed'].min(), 'to', td_raw['datetime_parsed'].max())
print('Unique dates:', td_raw['datetime_parsed'].dt.date.nunique())

# Check Timestamp numeric unique
td_raw['Timestamp'] = pd.to_numeric(td_raw['Timestamp'], errors='coerce')
print()
print('Numeric Timestamp unique:', td_raw['Timestamp'].nunique())
print('Numeric sample:', td_raw['Timestamp'].head(5).tolist())
