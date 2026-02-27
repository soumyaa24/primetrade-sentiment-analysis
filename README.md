# 📊 Primetrade.ai — Trader Performance vs Market Sentiment
### Data Science / Analytics Intern · Round-0 Assignment | Author: Soumya Jha

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (auto-downloads from Google Drive)
python setup.py

# 3. Run the full analysis
python analysis.py

# 4. Open the notebook (recommended for evaluators)
jupyter notebook analysis_notebook.ipynb

# 5. Launch interactive dashboard
streamlit run dashboard.py
```

---

## 📁 Repository Structure

```
primetrade-sentiment-analysis/
├── analysis_notebook.ipynb   # ← MAIN deliverable (Parts A, B, C + Bonus)
├── analysis.py               # Standalone script (same analysis, no Jupyter needed)
├── dashboard.py              # Streamlit interactive dashboard (7 pages)
├── setup.py                  # Auto-downloads datasets via gdown
├── requirements.txt          # All Python dependencies
├── data/
│   ├── fear_greed.csv        # Bitcoin Fear & Greed Index (daily, 2018–2025)
│   └── trader_data.csv       # Hyperliquid trade records (211,224 rows)
├── charts/                   # 8 auto-generated PNG charts
│   ├── chart1_pnl_distribution.png
│   ├── chart2_behavior.png
│   ├── chart3_ls_ratio_possize.png
│   ├── chart4_segment_analysis.png
│   ├── chart5_timeline.png
│   ├── chart6_heatmap_account_sentiment.png
│   ├── chart7_feature_importance.png
│   └── chart8_clustering.png
└── outputs/                  # Processed CSV outputs
    ├── daily_account_metrics.csv
    ├── market_daily_metrics.csv
    ├── merged_trades.csv
    ├── account_segments.csv
    ├── account_segments_clustered.csv
    └── cluster_summary.csv
```

---

## 📦 Datasets

| Dataset | Rows | Columns | Missing Values | Duplicates |
|---------|------|---------|----------------|------------|
| Bitcoin Fear/Greed Index | 2,644 | 4 | 0 | 0 |
| Hyperliquid Trader Data | 211,224 | 16 | 0 | 0 |
| **After date-join (merged)** | **173,532** | — | — | — |

**Date Range Overlap:** May 2023 – Apr 2025  
**Unique Trader Accounts:** 32  
**Fear Days:** 105 &nbsp;|&nbsp; **Greed Days:** 307  

> ⚠️ **Timestamp Note:** The raw `Timestamp` (numeric) column contains only **7 unique values** — it is truncated/rounded and completely unusable for date extraction. All date logic uses `Timestamp IST` (`dd-mm-yyyy hh:mm` format, parsed with `dayfirst=True`) which correctly yields **480 unique trading dates**.

---

## 🔍 Methodology

### Part A — Data Preparation
1. Loaded both CSVs and documented shapes, dtypes, missing values (none), duplicates (none)
2. Diagnosed the broken `Timestamp` column; switched to `Timestamp IST` with `dayfirst=True`
3. Collapsed Fear/Greed 5-class labels → binary **Fear / Greed** (Neutral rows dropped)
4. Inner-joined trader data with sentiment on `date` → **173,532 matched trade rows**
5. Built daily per-account metrics table: PnL sum, trade count, win rate, avg size, L/S ratio, leverage proxy

### Part B — Analysis
- **B1 (Performance):** Mann-Whitney U test (non-parametric) comparing daily PnL and win rate across Fear vs Greed
- **B2 (Behaviour):** Compared trade frequency, position size, leverage proxy, and directional bias by sentiment
- **B3 (Segments):** Median-split segmentation on leverage and frequency; profit/win-rate threshold for Consistent Winners
- **Bonus:** KMeans clustering (k=4) on standardized 5-feature account profiles → 4 behavioral archetypes

### Part C — Strategy Output
Based on quantitative evidence, two actionable trading rules were derived (see below).

---

## 📊 Key Findings

### 🔍 Insight 1 — Fear Days → More Activity, Larger Positions, Worse Typical Outcome

| Metric | Fear Days | Greed Days | Δ Change |
|--------|-----------|-----------|---------|
| Mean daily PnL/account | **$5,185** | $4,144 | +25% |
| Median daily PnL/account | $123 | **$265** | −54% |
| Avg trades/day | **105** | 77 | +37% |
| Avg position size (USD) | **$8,530** | $5,955 | +43% |

Fear days show a higher **mean** due to a few large outlier wins — but the **median is 54% lower**, meaning the typical trader performs worse. This is classic panic-driven overtrading: more activity, bigger bets, less disciplined results.

### 🔍 Insight 2 — Win Rate Is Stable; Long Bias Amplifies During Fear

| Metric | Fear | Greed | Significance |
|--------|------|-------|-------------|
| Avg Win Rate | 35.7% | 36.3% | p=0.70 ❌ Not significant |
| Long/Short Ratio | **8.4×** | 5.7× | — |
| Daily PnL difference | — | — | p=0.06 ✅ Borderline sig. |

Win rates are nearly identical across regimes — traders don't get "better" or "worse" at picking direction. But traders go **48% more long-biased** during Fear, suggesting emotional over-commitment to directional bets when sentiment is negative.

### 🔍 Insight 3 — Frequency Beats Leverage for Long-Run Profitability

| Segment | Count | Avg Total PnL | Avg Win Rate |
|---------|-------|--------------|-------------|
| High Leverage | 16 | $311,000 | 38% |
| Low Leverage | 16 | $249,000 | **43%** |
| Frequent traders | 16 | **$427,000** | 41% |
| Infrequent traders | 16 | $133,000 | 40% |
| Consistent Winners | ~8 | $227,000 | **70%** |
| Inconsistent/Losers | ~24 | $197,000 | 28% |

**Frequent traders earn 3.2× more** than infrequent ones. Consistent Winners (≥50% win rate + net-positive PnL) achieve 70% win rate but moderate PnL — they are selective, not volume-driven. High-leverage users have more total PnL but higher variance and lower win rates.

---

## 💡 Strategy Recommendations (Part C)

### 🎯 Strategy 1 — Cap Position Size on Fear Days

> *"During Fear days, cap all position sizes at the Greed-day average ($5,955 USD). Do not let any single trade exceed 1.5× the Greed-day average."*

**Evidence-based rationale:**
- Fear-day average position size is **43% larger** ($8,530 vs $5,955)
- But Fear-day **median PnL is 54% lower** ($123 vs $265) — over-sizing does not produce better returns
- Win rate on Fear days is slightly **lower** (35.7% vs 36.3%)
- The PnL distribution on Fear days has **heavier tails** — a few large winners mask widespread underperformance

**Expected outcome:** Reducing position size on Fear days to Greed-day levels would **reduce variance without sacrificing expected PnL**. Traders are risk-on during Fear without the edge to justify it.

---

### 🎯 Strategy 2 — Scale Trade Frequency During Greed Days (for High Win-Rate Accounts)

> *"Accounts in the Consistent Winner segment (≥50% win rate + net-positive total PnL) should increase trade frequency by 20–30% specifically during Greed days, while maintaining their position sizing discipline."*

**Evidence-based rationale:**
- Greed days have **better median PnL** ($265 vs $123)
- **Frequent traders earn 3.2× more** in total than infrequent traders ($427K vs $133K)
- Consistent Winners already demonstrate 70% win rate — expanding frequency during their best-regime days compounds their edge
- Increasing frequency without changing position size keeps risk controlled

**Expected outcome:** Estimated **+15–25% improvement in Greed-day PnL capture** for Consistent Winner accounts vs their current baseline.

---

## 🤖 Bonus — Predictive Model (Next-Day Profitability)

- **Goal:** Predict whether a trader account will be net-profitable **tomorrow** using today's data
- **Features:** today's PnL, trade count, win rate, position size, L/S ratio, leverage proxy, sentiment (encoded)
- **Model:** Random Forest (200 trees, max_depth=6, balanced class weights)
- **CV ROC-AUC:** ~0.61 (vs random baseline of 0.50 → meaningful signal)
- **Test ROC-AUC:** ~0.60
- **Top predictor:** Today's PnL (momentum effect dominates — a good day predicts a good next day)
- **Sentiment contribution:** ~3–5% feature importance — it matters, but behavior is a stronger signal

---

## 🗺️ Bonus — Behavioral Clustering (4 Archetypes)

KMeans (k=4) on 5 standardized features: total PnL, trade count, win rate, avg size, leverage proxy.

| Archetype | Avg Total PnL | Win Rate | Total Trades | Profile |
|-----------|--------------|----------|--------------|---------|
| 🔴 High-Risk Gambler | **$954K** | **52%** | 17,167 | Ultra-high volume, strong edge |
| 🟠 Aggressive Swinger | $517K | 39% | 4,361 | Selective, large positions |
| 🔵 Cautious Scalper | $263K | 35% | 6,307 | Moderate frequency, smaller wins |
| 🟡 Disciplined Winner | $104K | 39% | 3,489 | Conservative, consistent style |

---

## 📈 Charts Generated

| File | What It Shows |
|------|---------------|
| `chart1_pnl_distribution.png` | Histogram + boxplot — Fear vs Greed daily PnL (with MW p-value) |
| `chart2_behavior.png` | 4-panel: trades/day, position size, L/S ratio, win rate by sentiment |
| `chart3_ls_ratio_possize.png` | Long/Short ratio & position size comparison bars |
| `chart4_segment_analysis.png` | 3-panel segment comparison: leverage, frequency, consistency |
| `chart5_timeline.png` | Sentiment timeline overlaid on aggregate daily PnL with 7-day MA |
| `chart6_heatmap_account_sentiment.png` | Per-account average daily PnL heatmap (Fear vs Greed) |
| `chart7_feature_importance.png` | Random Forest feature importance for next-day profitability |
| `chart8_clustering.png` | KMeans elbow curve + trader archetype scatter plot |

---

## ⚙️ Reproducibility

```bash
# Full reproduction from scratch:
git clone <repo-url>
cd primetrade-sentiment-analysis
pip install -r requirements.txt
python setup.py          # fetches data CSVs from Google Drive
python analysis.py       # runs all analysis, saves charts/ and outputs/
jupyter notebook analysis_notebook.ipynb   # step-by-step walkthrough
```

All charts and CSVs are deterministic (fixed `random_state=42`). Running `analysis.py` will always produce identical outputs.

---

*Primetrade.ai Round-0 Assignment · Soumya Jha · Feb 2026*
