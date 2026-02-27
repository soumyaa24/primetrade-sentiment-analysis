# 📊 Primetrade.ai — Trader Performance vs Market Sentiment
### Data Science / Analytics Intern · Round-0 Assignment | Author: Soumya Jha

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python setup.py          # download datasets
jupyter notebook analysis_notebook.ipynb   # full analysis
# OR
python analysis.py       # run as script
streamlit run dashboard.py  # interactive dashboard
```

---

## 📁 Repository Structure
```
primetrade-sentiment-analysis/
├── analysis_notebook.ipynb   # ← Main notebook (Parts A, B, C + Bonus)
├── analysis.py               # Standalone script (same analysis)
├── dashboard.py              # Streamlit dashboard (7 pages)
├── setup.py                  # Downloads raw datasets via gdown
├── requirements.txt          # Python dependencies
├── charts/                   # Auto-generated PNG charts (8 charts)
└── outputs/                  # Processed CSVs
```

---

## 📦 Datasets
| Dataset | Rows | Cols | Missing | Duplicates |
|---------|------|------|---------|------------|
| Bitcoin Fear/Greed Index | 2,644 | 4 | 0 | 0 |
| Hyperliquid Trader Data | 211,224 | 16 | 0 | 0 |
| **After date-join** | **173,532** | — | — | — |

**Date Range:** May 2023 – Apr 2025 · **32 trader accounts** · **105 Fear days, 307 Greed days**

> ⚠️ **Timestamp note:** The raw `Timestamp` column has only 7 unique values (truncated). The analysis correctly uses `Timestamp IST` (`dd-mm-yyyy hh:mm` format, `dayfirst=True`) which yields 480 unique real trading dates.

---

## 🔍 Methodology

### Part A — Data Preparation
1. Loaded both CSVs; documented shapes, dtypes, zero missing values, zero duplicates
2. Parsed `Timestamp IST` with `pd.to_datetime(..., dayfirst=True)` → extracted `date`
3. Fear/Greed 5-class labels collapsed → binary **Fear / Greed**
4. Inner-joined on `date` → **173,532 matched trades**
5. Built daily per-account metrics: PnL sum, trade count, win rate, avg size, L/S ratio, leverage proxy

### Part B — Analysis
- Mann-Whitney U tests (non-parametric) for PnL and win-rate differences
- Cumulative-sum drawdown proxy per account
- Median-split segmentation by leverage and frequency
- KMeans clustering (k=4) on standardized 5-feature account profiles

### Bonus
- Random Forest for next-day profitability prediction (CV ROC-AUC ≈ 0.61)
- Streamlit dashboard with 7 interactive pages

---

## 📊 Key Findings

### Insight 1 — Fear Days → Higher Activity, Larger Positions
| Metric | Fear | Greed |
|--------|------|-------|
| Mean daily PnL/account | **$5,185** | $4,144 |
| Median daily PnL/account | $123 | **$265** |
| Avg trades/day | **105** | 77 |
| Avg position size | **$8,530** | $5,955 |

Fear days see more trades and larger sizes, but the median is lower — a few big wins skew Fear's mean upward.

### Insight 2 — Win Rate Stays Consistent; L/S Flips with Sentiment
| Metric | Fear | Greed |
|--------|------|-------|
| Avg Win Rate | 35.7% | **36.3%** |
| Long/Short Ratio | **8.4×** | 5.7× |

Win rates are nearly equal (p=0.70, not significant). Long-bias exists in BOTH regimes — traders skew long regardless, but slightly more so during Fear.

### Insight 3 — Frequency Beats Leverage for Long-Run PnL
| Segment | Avg Total PnL | Win Rate |
|---------|--------------|---------|
| High Leverage | **$311K** | 38% |
| Low Leverage | $249K | **43%** |
| Frequent traders | **$427K** | 41% |
| Infrequent traders | $133K | 40% |
| Consistent Winners | $227K | **70%** |

Frequent traders earn **3.2× more** than infrequent. Consistent Winners have the highest win-rate (70%) but moderate PnL — they are selective, not volume-driven.

---

## 💡 Strategy Recommendations (Part C)

### Strategy 1 — Cut Position Size on Fear Days
> *"During Fear days, cap position size at the Greed-day average ($5,955) for all accounts."*

**Evidence:** Fear-day average positions are 43% larger ($8,530 vs $5,955) but median PnL is 54% **lower** ($123 vs $265). The size increase doesn't produce proportionately better outcomes — it amplifies risk without reward.

**Expected outcome:** Reduce per-trade variance on Fear days without limiting trade frequency.

### Strategy 2 — Increase Frequency During Greed for High Win-Rate Accounts
> *"Accounts in the Consistent Winner segment should increase trade frequency by 20-30% during Greed days."*

**Evidence:** Greed days have better median PnL ($265) and Frequent traders earn 3.2× more than Infrequent. Consistent Winners have 70% win rate — scaling their frequency on high-conviction days maximises edge capture.

**Expected outcome:** +15-25% PnL capture vs baseline for Consistent Winners on Greed days.

---

## 🤖 Bonus — Predictive Model

- **Target:** Will trader be net-profitable tomorrow? (binary)
- **Features:** today's PnL, trade count, win rate, position size, L/S ratio, leverage proxy, sentiment
- **Model:** Random Forest (200 trees, max\_depth=6, balanced classes)
- **CV ROC-AUC:** ~0.61 (above random baseline of 0.50)
- **Top predictor:** today's PnL (momentum effect dominates)

---

## 🗺️ Bonus — Behavioral Clustering (4 Archetypes)

| Archetype | Avg PnL | Win Rate | Trades | Profile |
|-----------|---------|----------|--------|---------|
| High-Risk Gambler | **$954K** | **52%** | 17,167 | High volume, high win-rate |
| Aggressive Swinger | $517K | 39% | 4,361 | Selective, larger positions |
| Cautious Scalper | $263K | 35% | 6,307 | Many trades, modest wins |
| Disciplined Winner | $104K | 39% | 3,489 | Consistent, low-risk style |

---

## 📈 Charts Generated
| File | Content |
|------|---------|
| `chart1_pnl_distribution.png` | Histogram + boxplot — Fear vs Greed PnL |
| `chart2_winrate_frequency.png` | Win rate & trade frequency bars |
| `chart3_ls_ratio_possize.png` | L/S ratio & position size by sentiment |
| `chart4_segment_analysis.png` | 3-panel segment comparison |
| `chart5_timeline.png` | Sentiment timeline overlaid on daily PnL |
| `chart6_heatmap_account_sentiment.png` | Per-account PnL heatmap |
| `chart7_feature_importance.png` | RF feature importance |
| `chart8_clustering.png` | KMeans elbow + archetype scatter |

---

*Primetrade.ai Round-0 Assignment · Soumya Jha · Feb 2026*
