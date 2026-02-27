import pandas as pd, numpy as np
from datetime import datetime

td_raw = pd.read_csv('data/trader_data.csv')
fg_raw = pd.read_csv('data/fear_greed.csv')

print('=== TRADER DATA TIMESTAMP DEBUG ===')
print('Timestamp IST col sample:')
for v in td_raw['Timestamp IST'].head(10).tolist():
    print(' ', repr(v))
print()
print('Timestamp (numeric) sample:')
for v in td_raw['Timestamp'].head(5).tolist():
    print(' ', v)

# Try parsing Timestamp IST
print()
print('Testing pd.to_datetime on Timestamp IST...')
try:
    ts1 = pd.to_datetime(td_raw['Timestamp IST'].head(3), dayfirst=True)
    print(ts1.tolist())
except Exception as e:
    print('Error:', e)

# Numeric timestamp range
ts_num = pd.to_numeric(td_raw['Timestamp'], errors='coerce').dropna()
print()
print('Numeric Timestamp min:', ts_num.min())
print('Numeric Timestamp max:', ts_num.max())
# Check if seconds
if ts_num.max() < 2e10:
    print('Interpretation: Unix seconds')
    print('Min date:', datetime.fromtimestamp(ts_num.min()))
    print('Max date:', datetime.fromtimestamp(ts_num.max()))
elif ts_num.max() < 2e13:
    print('Interpretation: Unix milliseconds')
    print('Min date:', datetime.fromtimestamp(ts_num.min()/1000))
    print('Max date:', datetime.fromtimestamp(ts_num.max()/1000))

print()
print('FG date sample:')
for v in fg_raw['date'].head(5).tolist():
    print(' ', repr(v))
print('FG classification unique:', fg_raw['classification'].unique().tolist())
