"""
setup.py – Downloads the raw datasets from Google Drive.
Run once before the analysis: python setup.py
"""
import os
import gdown

os.makedirs("data", exist_ok=True)

print("Downloading Bitcoin Fear/Greed Index...")
gdown.download(
    "https://drive.google.com/uc?id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf",
    "data/fear_greed.csv",
    quiet=False
)

print("Downloading Hyperliquid Trader Data...")
gdown.download(
    "https://drive.google.com/uc?id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs",
    "data/trader_data.csv",
    quiet=False
)

print("\nDone! Both datasets saved to data/")
print("Next step: python analysis.py")
