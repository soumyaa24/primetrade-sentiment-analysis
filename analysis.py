"""
Primetrade.ai – Trader Performance vs Market Sentiment
=======================================================
CORRECTED VERSION: Uses 'Timestamp IST' (dd-mm-yyyy format) for date extraction
Author : Soumya Jha
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor":  "#58a6ff", "xtick.color": "#8b949e",
    "ytick.color":      "#8b949e", "text.color": "#c9d1d9",
    "grid.color":       "#21262d", "grid.alpha": 0.6,
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "figure.dpi":       150,
})

FEAR_COLOR  = "#ff453a"
GREED_COLOR = "#30d158"

os.makedirs("charts",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("="*70)
print("  PRIMETRADE.AI – TRADER PERFORMANCE VS MARKET SENTIMENT (v2)")
print("="*70)

# ==============================================================================
# PART A – DATA PREPARATION
# ==============================================================================
print("\n[A] DATA PREPARATION\n")

# ── Load ──────────────────────────────────────────────────────────────────────
fg_raw = pd.read_csv("data/fear_greed.csv")
td_raw = pd.read_csv("data/trader_data.csv")

print(f"  Fear/Greed raw  : {fg_raw.shape[0]:,} rows x {fg_raw.shape[1]} cols")
print(f"  Trader data raw : {td_raw.shape[0]:,} rows x {td_raw.shape[1]} cols")

# ── Missing & Duplicates ──────────────────────────────────────────────────────
print("\n  [A1] Missing values:")
print("  Fear/Greed:\n", fg_raw.isnull().sum().to_string())
print("  Trader data:\n", td_raw.isnull().sum().to_string())
print(f"\n  Duplicates -> Fear/Greed: {fg_raw.duplicated().sum()}  | Trader: {td_raw.duplicated().sum()}")

# ── Clean Fear/Greed ──────────────────────────────────────────────────────────
fg = fg_raw.copy()
fg["classification"] = fg["classification"].str.strip()
fg["sentiment"] = fg["classification"].apply(
    lambda x: "Fear" if "Fear" in str(x) else ("Greed" if "Greed" in str(x) else "Neutral")
)
fg["date"] = pd.to_datetime(fg["date"])
fg = fg[fg["sentiment"].isin(["Fear","Greed"])]   # keep only Fear/Greed
fg = fg.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
print(f"\n  FG cleaned: {len(fg)} days | range: {fg.date.min().date()} to {fg.date.max().date()}")
print("  Sentiment dist:", fg.sentiment.value_counts().to_dict())

# ── Clean Trader Data (USE Timestamp IST — dayfirst dd-mm-yyyy) ────────────────
td = td_raw.copy()
td["datetime"] = pd.to_datetime(td["Timestamp IST"], dayfirst=True, errors="coerce")
parse_errors = td["datetime"].isna().sum()
print(f"\n  Timestamp IST parse errors: {parse_errors}")

td = td.dropna(subset=["datetime"])
td["date"] = td["datetime"].dt.normalize()   # midnight date

# Coerce numeric columns
for col in ["Execution Price","Size Tokens","Size USD","Closed PnL",
            "Start Position","Fee","Trade ID"]:
    td[col] = pd.to_numeric(td[col], errors="coerce")

# Keep only rows with PnL (actual trades, not fund transfers)
td_trades = td[td["Closed PnL"].notna()].copy()

print(f"  Trader date range: {td_trades.date.min().date()} to {td_trades.date.max().date()}")
print(f"  Unique accounts  : {td_trades.Account.nunique()}")
print(f"  Unique dates     : {td_trades.date.nunique()}")
print(f"  Trade rows w/ PnL: {len(td_trades):,}")

# ── Merge on date ─────────────────────────────────────────────────────────────
merged = td_trades.merge(
    fg[["date","sentiment","value","classification"]],
    on="date", how="inner"
)
print(f"\n  After merge: {len(merged):,} rows")
print(f"  Date overlap: {merged.date.min().date()} to {merged.date.max().date()}")
print(f"  Sentiment in merged:\n{merged.sentiment.value_counts().to_string()}")

# ── Key Metrics ───────────────────────────────────────────────────────────────
print("\n  [A3] Computing key metrics ...")

merged["is_win"] = (merged["Closed PnL"] > 0).astype(int)
merged["lev_proxy"] = np.where(
    merged["Start Position"].abs() > 0,
    merged["Size USD"].abs() / (merged["Start Position"].abs() + 1e-9),
    np.nan
)

# Daily account-level aggregation
daily = (merged.groupby(["Account","date","sentiment"])
               .agg(
                    daily_pnl    = ("Closed PnL", "sum"),
                    n_trades     = ("Closed PnL", "count"),
                    win_count    = ("is_win", "sum"),
                    avg_size_usd = ("Size USD", "mean"),
               ).reset_index())
daily["win_rate"] = daily["win_count"] / daily["n_trades"]

# Long/Short ratio
sides = (merged.groupby(["Account","date","sentiment"])
               .apply(lambda g: (g["Side"]=="BUY").sum() / max((g["Side"]=="SELL").sum(), 1))
               .reset_index(name="long_short_ratio"))
daily = daily.merge(sides, on=["Account","date","sentiment"], how="left")

# Leverage proxy
lev_d = (merged.groupby(["Account","date"])["lev_proxy"]
               .median().reset_index(name="median_lev_proxy"))
daily = daily.merge(lev_d, on=["Account","date"], how="left")

print(f"  Daily rows: {len(daily):,} across {daily.date.nunique()} unique dates")
print(f"  Sentiment distribution in daily:\n{daily.groupby('sentiment').size().to_string()}")

# Market-level daily view
mkt_daily = (daily.groupby(["date","sentiment"])
                  .agg(market_daily_pnl=("daily_pnl","sum"),
                       avg_win_rate=("win_rate","mean"),
                       avg_n_trades=("n_trades","mean"),
                       avg_size_usd=("avg_size_usd","mean"),
                       avg_ls_ratio=("long_short_ratio","mean"))
                  .reset_index())

# Save outputs
daily.to_csv("outputs/daily_account_metrics.csv", index=False)
mkt_daily.to_csv("outputs/market_daily_metrics.csv", index=False)
merged.to_csv("outputs/merged_trades.csv", index=False)
print("  Saved outputs/daily_account_metrics.csv, market_daily_metrics.csv, merged_trades.csv")

# ==============================================================================
# PART B – ANALYSIS
# ==============================================================================
print("\n"+"="*70)
print("[B] ANALYSIS")
print("="*70)

fear_d  = daily[daily["sentiment"] == "Fear"]
greed_d = daily[daily["sentiment"] == "Greed"]

# ── B1: PnL / Win Rate / Drawdown ─────────────────────────────────────────────
print("\n  [B1] PnL, Win Rate, Drawdown – Fear vs Greed")

pnl_fear  = fear_d["daily_pnl"].dropna()
pnl_greed = greed_d["daily_pnl"].dropna()
wr_fear   = fear_d["win_rate"].dropna()
wr_greed  = greed_d["win_rate"].dropna()

_, p_pnl = stats.mannwhitneyu(pnl_fear, pnl_greed, alternative="two-sided")
_, p_wr  = stats.mannwhitneyu(wr_fear,  wr_greed,  alternative="two-sided")

print(f"    Daily PnL  | Fear  mean={pnl_fear.mean():.2f}  median={pnl_fear.median():.2f}  std={pnl_fear.std():.2f}")
print(f"    Daily PnL  | Greed mean={pnl_greed.mean():.2f}  median={pnl_greed.median():.2f}  std={pnl_greed.std():.2f}")
print(f"    MW p-value (PnL)     : {p_pnl:.4f}  {'*significant*' if p_pnl<0.05 else '(not significant)'}")
print(f"    Win Rate   | Fear  {wr_fear.mean():.4f}  Greed {wr_greed.mean():.4f}")
print(f"    MW p-value (WinRate) : {p_wr:.4f}  {'*significant*' if p_wr<0.05 else '(not significant)'}")

def max_drawdown(series):
    cs = series.cumsum()
    return (cs - cs.cummax()).min()

dd_fear  = fear_d.groupby("Account")["daily_pnl"].apply(max_drawdown)
dd_greed = greed_d.groupby("Account")["daily_pnl"].apply(max_drawdown)
print(f"    Drawdown proxy | Fear: {dd_fear.mean():.2f}  Greed: {dd_greed.mean():.2f}")

# ── B2: Behavioral Differences ────────────────────────────────────────────────
print("\n  [B2] Behavioral Differences")
for metric, label in [("n_trades","Avg Trades/Day"),("avg_size_usd","Avg Size USD"),
                       ("long_short_ratio","Long/Short Ratio"),("median_lev_proxy","Median Lev Proxy")]:
    fv = fear_d[metric].dropna().mean()
    gv = greed_d[metric].dropna().mean()
    print(f"    {label:<25} Fear={fv:.3f}  Greed={gv:.3f}  delta={gv-fv:+.3f}")

# ── B3: Segments ──────────────────────────────────────────────────────────────
print("\n  [B3] Trader Segments")

acct = (merged.groupby("Account")
              .agg(total_pnl    = ("Closed PnL","sum"),
                   n_trades     = ("Closed PnL","count"),
                   win_rate     = ("is_win","mean"),
                   avg_size_usd = ("Size USD","mean"),
                   med_lev      = ("lev_proxy","median"))
              .reset_index())

lev_med   = acct["med_lev"].median()
trade_med = acct["n_trades"].median()
acct["lev_seg"]    = np.where(acct["med_lev"]   >= lev_med,   "High Leverage",      "Low Leverage")
acct["freq_seg"]   = np.where(acct["n_trades"]  >= trade_med, "Frequent",           "Infrequent")
acct["winner_seg"] = np.where((acct["total_pnl"]>0)&(acct["win_rate"]>=0.5),
                               "Consistent Winner","Inconsistent/Loser")

print("  Segment summary:")
for seg_col in ["lev_seg","freq_seg","winner_seg"]:
    grp = acct.groupby(seg_col)[["total_pnl","win_rate","n_trades"]].mean().round(2)
    print(f"\n  [{seg_col}]")
    print(grp.to_string())

acct.to_csv("outputs/account_segments.csv", index=False)
print("\n  Saved: outputs/account_segments.csv")

# ==============================================================================
# CHARTS
# ==============================================================================
print("\n  [Charts] Generating ...")

# Chart 1: PnL distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 1 -- Daily PnL Distribution: Fear vs Greed", fontsize=14, fontweight="bold", color="#58a6ff")
clip_val = max(abs(pnl_fear.quantile(0.95)), abs(pnl_greed.quantile(0.95))) * 1.5

ax = axes[0]
ax.hist(pnl_fear.clip(-clip_val, clip_val),  bins=60, color=FEAR_COLOR,  alpha=0.75, label="Fear")
ax.hist(pnl_greed.clip(-clip_val, clip_val), bins=60, color=GREED_COLOR, alpha=0.6,  label="Greed")
ax.axvline(pnl_fear.median(),  color=FEAR_COLOR,  ls="--", lw=1.5, label=f"Fear median={pnl_fear.median():.0f}")
ax.axvline(pnl_greed.median(), color=GREED_COLOR, ls="--", lw=1.5, label=f"Greed median={pnl_greed.median():.0f}")
ax.set_xlabel("Daily PnL (USD)"); ax.set_ylabel("Frequency")
ax.set_title("Histogram"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

bp = axes[1].boxplot([pnl_fear.clip(-clip_val, clip_val), pnl_greed.clip(-clip_val, clip_val)],
                     patch_artist=True, labels=["Fear","Greed"],
                     medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], [FEAR_COLOR, GREED_COLOR]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1].set_ylabel("Daily PnL (USD)"); axes[1].set_title(f"Box Plot  (MW p={p_pnl:.4f})")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("charts/chart1_pnl_distribution.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart1_pnl_distribution.png")

# Chart 2: Win Rate & Trade Frequency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Chart 2 -- Win Rate & Trade Frequency: Fear vs Greed", fontsize=14, fontweight="bold", color="#58a6ff")
for ax, vals, title, ylabel in [
    (axes[0], [wr_fear.mean(), wr_greed.mean()], "Average Win Rate", "Win Rate"),
    (axes[1], [fear_d["n_trades"].mean(), greed_d["n_trades"].mean()], "Avg Trades/Day per Account", "Trades"),
]:
    bars = ax.bar(["Fear","Greed"], vals, color=[FEAR_COLOR, GREED_COLOR], alpha=0.85, width=0.5)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.3f}", ha="center", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/chart2_winrate_frequency.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart2_winrate_frequency.png")

# Chart 3: L/S Ratio & Position Size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Chart 3 -- Long/Short Bias & Position Size: Fear vs Greed", fontsize=14, fontweight="bold", color="#58a6ff")
ls_vals   = [fear_d["long_short_ratio"].mean(), greed_d["long_short_ratio"].mean()]
size_vals = [fear_d["avg_size_usd"].mean(), greed_d["avg_size_usd"].mean()]

for ax, vals, title, ylabel in [
    (axes[0], ls_vals,   "Long/Short Ratio", "Ratio"),
    (axes[1], size_vals, "Avg Position Size (USD)", "USD"),
]:
    bars = ax.bar(["Fear","Greed"], vals, color=[FEAR_COLOR, GREED_COLOR], alpha=0.85, width=0.5)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.3)
    if title.startswith("Long"):
        ax.axhline(1.0, color="white", ls="--", lw=1, alpha=0.5, label="Neutral=1.0"); ax.legend(fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.2f}", ha="center", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/chart3_ls_ratio_possize.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart3_ls_ratio_possize.png")

# Chart 4: Segment analysis
palette = {"High Leverage":"#ff6b6b","Low Leverage":"#48dbfb",
           "Frequent":"#ff9f43","Infrequent":"#a29bfe",
           "Consistent Winner":"#6ab04c","Inconsistent/Loser":"#eb4d4b"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Chart 4 -- Trader Segment Analysis", fontsize=14, fontweight="bold", color="#58a6ff")

for ax, (seg_col, y_col, ylabel, title) in zip(axes, [
    ("lev_seg",    "total_pnl", "Avg Total PnL (USD)", "High vs Low Leverage"),
    ("freq_seg",   "total_pnl", "Avg Total PnL (USD)", "Frequent vs Infrequent"),
    ("winner_seg", "win_rate",  "Avg Win Rate",         "Winners vs Losers"),
]):
    grp = acct.groupby(seg_col)[y_col].mean().reset_index()
    bc  = [palette.get(s, "#888") for s in grp[seg_col]]
    bars = ax.bar(grp[seg_col], grp[y_col], color=bc, alpha=0.85)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
    for b in bars:
        v = b.get_height()
        lbl = f"${v:,.0f}" if "PnL" in ylabel else f"{v:.3f}"
        ax.text(b.get_x()+b.get_width()/2, v*1.02 if v>=0 else v*0.98,
                lbl, ha="center", fontsize=8, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/chart4_segment_analysis.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart4_segment_analysis.png")

# Chart 5: Sentiment timeline
mkt_ts = mkt_daily.sort_values("date").copy()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                gridspec_kw={"height_ratios": [1, 2]})
fig.suptitle("Chart 5 -- Sentiment Timeline vs Aggregate Daily PnL", fontsize=14, fontweight="bold", color="#58a6ff")

for _, row in mkt_ts.iterrows():
    c = FEAR_COLOR if row["sentiment"] == "Fear" else GREED_COLOR
    ax1.axvspan(row["date"] - pd.Timedelta(hours=12),
                row["date"] + pd.Timedelta(hours=12), color=c, alpha=0.7)
ax1.set_ylabel("Sentiment"); ax1.set_yticks([])
ax1.legend(handles=[mpatches.Patch(color=FEAR_COLOR, label="Fear"),
                    mpatches.Patch(color=GREED_COLOR, label="Greed")], loc="upper right", fontsize=8)

pos_mask = mkt_ts["market_daily_pnl"] >= 0
ax2.fill_between(mkt_ts["date"], mkt_ts["market_daily_pnl"], 0,
                 where=pos_mask, color=GREED_COLOR, alpha=0.6, label="Positive PnL")
ax2.fill_between(mkt_ts["date"], mkt_ts["market_daily_pnl"], 0,
                 where=~pos_mask, color=FEAR_COLOR, alpha=0.6, label="Negative PnL")
ax2.plot(mkt_ts["date"], mkt_ts["market_daily_pnl"].rolling(7, min_periods=1).mean(),
         color="white", lw=1.5, label="7-day MA", alpha=0.9)
ax2.set_ylabel("Aggregate Daily PnL (USD)"); ax2.set_xlabel("Date")
ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("charts/chart5_timeline.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart5_timeline.png")

# Chart 6: Heatmap
pivot = daily.groupby(["Account","sentiment"])["daily_pnl"].mean().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, max(5, len(pivot)*0.55)))
sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax, annot=True, fmt=".0f",
            linewidths=0.4, annot_kws={"size": 8})
ax.set_title("Chart 6 -- Average Daily PnL per Account by Sentiment", color="#58a6ff", pad=12)
plt.tight_layout()
plt.savefig("charts/chart6_heatmap_account_sentiment.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart6_heatmap_account_sentiment.png")

# ==============================================================================
# BONUS: Predictive Model
# ==============================================================================
print("\n"+"="*70)
print("[BONUS] PREDICTIVE MODEL")
print("="*70)

daily_sorted = daily.sort_values(["Account","date"]).copy()
daily_sorted["next_pnl"]        = daily_sorted.groupby("Account")["daily_pnl"].shift(-1)
daily_sorted["next_profitable"] = (daily_sorted["next_pnl"] > 0).astype(int)

feat_cols = ["daily_pnl","n_trades","win_rate","avg_size_usd","long_short_ratio","median_lev_proxy"]
model_df  = daily_sorted[feat_cols + ["next_profitable","sentiment"]].dropna().copy()
le = LabelEncoder()
model_df["sentiment_enc"] = le.fit_transform(model_df["sentiment"])
feat_final = feat_cols + ["sentiment_enc"]

X = model_df[feat_final].values
y = model_df["next_profitable"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X_tr, y_tr)

y_pred  = rf.predict(X_te)
y_proba = rf.predict_proba(X_te)[:, 1]
cv_auc  = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()

print(f"  Class dist  : {pd.Series(y).value_counts().to_dict()}")
print(f"  Test ROC-AUC: {roc_auc_score(y_te, y_proba):.4f}")
print(f"  CV ROC-AUC  : {cv_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_te, y_pred, target_names=["Not Profitable","Profitable"]))

fi = pd.DataFrame({"Feature":feat_final,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
ax.barh(fi["Feature"], fi["Importance"],
        color=plt.cm.plasma(np.linspace(0.2, 0.9, len(fi))), alpha=0.9)
ax.set_xlabel("Feature Importance")
ax.set_title("Chart 7 -- RF Feature Importance (Next-day Profitability)", color="#58a6ff")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("charts/chart7_feature_importance.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart7_feature_importance.png")

# ==============================================================================
# BONUS: Clustering
# ==============================================================================
print("\n"+"="*70)
print("[BONUS] BEHAVIORAL CLUSTERING")
print("="*70)

cluster_feats = ["total_pnl","n_trades","win_rate","avg_size_usd","med_lev"]
acct_clean    = acct[cluster_feats].fillna(acct[cluster_feats].median())

scaler  = StandardScaler()
X_clust = scaler.fit_transform(acct_clean)

inertias = []
K_range  = range(2, min(8, len(acct)+1))
for k in K_range:
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_clust).inertia_)

best_k   = min(4, len(acct))
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
acct["cluster"]   = km_final.fit_predict(X_clust)
arch_map  = {0:"Cautious Scalper", 1:"Aggressive Swinger", 2:"Disciplined Winner", 3:"High-Risk Gambler"}
acct["archetype"] = acct["cluster"].map(arch_map)

print("  Archetype summary:")
print(acct.groupby("archetype")[cluster_feats].mean().round(2).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 8 -- Behavioral Clustering", fontsize=14, fontweight="bold", color="#58a6ff")

ax1.plot(list(K_range), inertias, marker="o", color="#58a6ff", lw=2, ms=8, mfc="#ff453a")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve"); ax1.grid(alpha=0.3)

clr = ["#ff453a","#30d158","#0a84ff","#ffd60a"]
for cid, arch in arch_map.items():
    sub = acct[acct["cluster"]==cid]
    ax2.scatter(sub["n_trades"], sub["total_pnl"], c=clr[cid], s=180,
                alpha=0.85, edgecolors="white", lw=0.5, label=arch, zorder=5)
    for _, row in sub.iterrows():
        ax2.annotate(row["Account"][:6]+"...", (row["n_trades"], row["total_pnl"]),
                     textcoords="offset points", xytext=(5,3), fontsize=6, color="#c9d1d9", alpha=0.7)
ax2.set_xlabel("Total Trades"); ax2.set_ylabel("Total PnL (USD)")
ax2.set_title("Trader Archetypes"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("charts/chart8_clustering.png", bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  Saved: charts/chart8_clustering.png")

acct.to_csv("outputs/account_segments_clustered.csv", index=False)
acct.groupby("archetype")[cluster_feats].mean().round(2).to_csv("outputs/cluster_summary.csv")
print("  Saved: outputs/account_segments_clustered.csv, cluster_summary.csv")

# ==============================================================================
# FINAL NUMBERS FOR WRITE-UP
# ==============================================================================
print("\n"+"="*70)
print("[RESULTS SUMMARY]")
print("="*70)
print(f"  Total merged trades       : {len(merged):,}")
print(f"  Date overlap              : {merged.date.min().date()} to {merged.date.max().date()}")
print(f"  Unique accounts           : {merged.Account.nunique()}")
print(f"  Fear days (dates)         : {daily[daily.sentiment=='Fear']['date'].nunique()}")
print(f"  Greed days (dates)        : {daily[daily.sentiment=='Greed']['date'].nunique()}")
print(f"  Fear account-day rows     : {len(fear_d)}")
print(f"  Greed account-day rows    : {len(greed_d)}")
print(f"  Mean daily PnL - Fear     : {pnl_fear.mean():.2f}")
print(f"  Mean daily PnL - Greed    : {pnl_greed.mean():.2f}")
print(f"  Median daily PnL - Fear   : {pnl_fear.median():.2f}")
print(f"  Median daily PnL - Greed  : {pnl_greed.median():.2f}")
print(f"  Win Rate - Fear           : {wr_fear.mean():.4f}")
print(f"  Win Rate - Greed          : {wr_greed.mean():.4f}")
print(f"  Avg trades/day - Fear     : {fear_d.n_trades.mean():.2f}")
print(f"  Avg trades/day - Greed    : {greed_d.n_trades.mean():.2f}")
print(f"  L/S ratio - Fear          : {fear_d.long_short_ratio.mean():.3f}")
print(f"  L/S ratio - Greed         : {greed_d.long_short_ratio.mean():.3f}")
print(f"  MW p-value PnL            : {p_pnl:.4f}")
print(f"  MW p-value WinRate        : {p_wr:.4f}")
print(f"  CV ROC-AUC (model)        : {cv_auc:.4f}")
print("\n  ANALYSIS COMPLETE")
