import pandas as pd, numpy as np
from scipy import stats

daily  = pd.read_csv('outputs/daily_account_metrics.csv')
acct   = pd.read_csv('outputs/account_segments_clustered.csv')
mkt    = pd.read_csv('outputs/market_daily_metrics.csv')
merged = pd.read_csv('outputs/merged_trades.csv')

fear  = daily[daily['sentiment']=='Fear']
greed = daily[daily['sentiment']=='Greed']

lines = []
lines.append('=== CORRECTED RESULTS ===')
lines.append(f'Total merged trades: {len(merged):,}')
lines.append(f'Date range: {merged.date.min()} to {merged.date.max()}')
lines.append(f'Unique accounts: {merged.Account.nunique()}')
lines.append(f'Fear account-day rows: {len(fear)}')
lines.append(f'Greed account-day rows: {len(greed)}')
lines.append(f'Fear unique dates: {fear.date.nunique()}')
lines.append(f'Greed unique dates: {greed.date.nunique()}')
lines.append('')
lines.append(f'Mean daily PnL Fear:  {fear.daily_pnl.mean():.2f}')
lines.append(f'Mean daily PnL Greed: {greed.daily_pnl.mean():.2f}')
lines.append(f'Median daily PnL Fear:  {fear.daily_pnl.median():.2f}')
lines.append(f'Median daily PnL Greed: {greed.daily_pnl.median():.2f}')
lines.append(f'Win Rate Fear:  {fear.win_rate.mean():.4f}')
lines.append(f'Win Rate Greed: {greed.win_rate.mean():.4f}')
lines.append(f'Avg trades Fear:  {fear.n_trades.mean():.2f}')
lines.append(f'Avg trades Greed: {greed.n_trades.mean():.2f}')
lines.append(f'L/S ratio Fear:  {fear.long_short_ratio.mean():.3f}')
lines.append(f'L/S ratio Greed: {greed.long_short_ratio.mean():.3f}')
lines.append(f'Avg size USD Fear:  {fear.avg_size_usd.mean():.2f}')
lines.append(f'Avg size USD Greed: {greed.avg_size_usd.mean():.2f}')
_, p_pnl = stats.mannwhitneyu(fear.daily_pnl.dropna(), greed.daily_pnl.dropna(), alternative='two-sided')
_, p_wr  = stats.mannwhitneyu(fear.win_rate.dropna(),  greed.win_rate.dropna(),  alternative='two-sided')
lines.append(f'MW p PnL:     {p_pnl:.4f}')
lines.append(f'MW p WinRate: {p_wr:.4f}')
lines.append('')
lines.append('=== SEGMENT SUMMARY ===')
for seg in ['lev_seg','freq_seg','winner_seg','archetype']:
    grp = acct.groupby(seg)[['total_pnl','win_rate','n_trades']].mean().round(2)
    lines.append(f'\n[{seg}]:')
    lines.append(grp.to_string())

with open('outputs/final_numbers.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('written')
