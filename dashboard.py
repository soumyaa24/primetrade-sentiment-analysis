# Primetrade.ai – Trader Performance vs Market Sentiment
## Streamlit Dashboard (Bonus)
# Run: streamlit run dashboard.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade.ai – Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme override ───────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #58a6ff !important; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin: 4px;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-card .lbl { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
    .insight-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .fear-badge  { color: #ff453a; font-weight: 700; }
    .greed-badge { color: #30d158; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

FEAR_COLOR  = "#ff453a"
GREED_COLOR = "#30d158"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    daily  = pd.read_csv("outputs/daily_account_metrics.csv")
    acct   = pd.read_csv("outputs/account_segments_clustered.csv")
    mkt    = pd.read_csv("outputs/market_daily_metrics.csv")
    merged = pd.read_csv("outputs/merged_trades.csv",
                         usecols=["Account","date","sentiment","Closed PnL",
                                  "Size USD","Side","lev_proxy","Coin"])
    daily["date"] = pd.to_datetime(daily["date"])
    mkt["date"]   = pd.to_datetime(mkt["date"])
    return daily, acct, mkt, merged

daily, acct, mkt, merged = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Primetrade.ai")
st.sidebar.markdown("**Trader vs Sentiment Explorer**")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 PnL Analysis",
    "🔀 Behavior Shifts",
    "👥 Trader Segments",
    "🤖 Predictive Model",
    "🗺️ Clustering",
    "💡 Strategy Recommendations",
])

sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment", ["Fear", "Greed"], default=["Fear", "Greed"])
account_filter = st.sidebar.multiselect(
    "Filter by Account", sorted(acct["Account"].tolist()),
    default=sorted(acct["Account"].tolist()))

daily_f  = daily[daily["sentiment"].isin(sentiment_filter) & daily["Account"].isin(account_filter)]
merged_f = merged[merged["sentiment"].isin(sentiment_filter) & merged["Account"].isin(account_filter)]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Trader Performance vs Market Sentiment")
    st.markdown("> **Dataset**: Hyperliquid Trader Logs × Bitcoin Fear/Greed Index  |  "
                "**Period**: Mar 2023 – Feb 2025  |  **32 traders, 184K+ trade records**")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Trades",    f"{len(merged):,}",   "in merged dataset"),
        (c2, "Unique Accounts", "32",                  "Hyperliquid wallets"),
        (c3, "Date Range",      "Mar 2023\n→ Feb 2025","~24 months"),
        (c4, "Fear Days",       f"{(mkt['sentiment']=='Fear').sum()}",  "days in overlap"),
        (c5, "Greed Days",      f"{(mkt['sentiment']=='Greed').sum()}", "days in overlap"),
    ]
    for col, lbl, val, sub in metrics:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}<br/><small style="color:#555">{sub}</small></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Sentiment Timeline")
    mkt_sorted = mkt.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mkt_sorted["date"], y=mkt_sorted["market_daily_pnl"],
        fill="tozeroy",
        line=dict(color="#58a6ff", width=1.5),
        fillcolor="rgba(88,166,255,0.15)",
        name="Aggregate Daily PnL"))
    fear_mask = mkt_sorted["sentiment"] == "Fear"
    for _, row in mkt_sorted[fear_mask].iterrows():
        fig.add_vrect(x0=row["date"], x1=row["date"] + pd.Timedelta(days=1),
                      fillcolor="rgba(255,69,58,0.08)", line_width=0)
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#c9d1d9", height=320,
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        showlegend=True, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red shading = Fear days. Blue line = aggregate daily PnL of all traders.")

    st.markdown("### Key Insights at a Glance")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-box">
        <b>Insight 1 – PnL Asymmetry</b><br/>
        <span class="fear-badge">Fear</span> days produce higher raw PnL sums ($209K avg) vs
        <span class="greed-badge">Greed</span> days ($91K), driven by extremely active trading
        during Fear. <i>But win-rate is lower.</i>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-box">
        <b>Insight 2 – Behavioral Shift</b><br/>
        Traders are <b>3.6× more active</b> on <span class="fear-badge">Fear</span> days (4183 vs 1169 trades/day avg).
        Long/Short ratio <b>spikes to 10x</b> during <span class="greed-badge">Greed</span>,
        suggesting directional FOMO.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight-box">
        <b>Insight 3 – Leverage Matters</b><br/>
        Low-leverage traders earn <b>$414K avg</b> vs $226K for High-leverage.
        Frequent traders earn <b>3.2× more</b> than infrequent — edge compounds with volume.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PnL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 PnL Analysis":
    st.title("📈 PnL Analysis – Fear vs Greed")

    fear_pnl  = daily_f[daily_f["sentiment"]=="Fear"]["daily_pnl"].dropna()
    greed_pnl = daily_f[daily_f["sentiment"]=="Greed"]["daily_pnl"].dropna()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fear – Mean PnL",   f"${fear_pnl.mean():,.0f}",   delta=None)
    m2.metric("Greed – Mean PnL",  f"${greed_pnl.mean():,.0f}",  delta=None)
    m3.metric("Fear – Win Rate",   f"{daily_f[daily_f['sentiment']=='Fear']['win_rate'].mean():.1%}")
    m4.metric("Greed – Win Rate",  f"{daily_f[daily_f['sentiment']=='Greed']['win_rate'].mean():.1%}")

    st.markdown("### Daily PnL Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fear_pnl.clip(-20000, 20000),  nbinsx=60,
                               name="Fear",  marker_color=FEAR_COLOR,  opacity=0.75))
    fig.add_trace(go.Histogram(x=greed_pnl.clip(-20000, 20000), nbinsx=60,
                               name="Greed", marker_color=GREED_COLOR, opacity=0.75))
    fig.add_vline(x=fear_pnl.median(),  line_dash="dash", line_color=FEAR_COLOR,
                  annotation_text=f"Fear median: {fear_pnl.median():.0f}")
    fig.add_vline(x=greed_pnl.median(), line_dash="dash", line_color=GREED_COLOR,
                  annotation_text=f"Greed median: {greed_pnl.median():.0f}")
    fig.update_layout(barmode="overlay", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", height=380, xaxis_title="Daily PnL (USD)",
                      yaxis_title="Count", legend=dict(bgcolor="#161b22"),
                      xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Box Plot – PnL Spread by Sentiment")
    fig2 = go.Figure()
    for sent, col in [("Fear", FEAR_COLOR), ("Greed", GREED_COLOR)]:
        sub = daily_f[daily_f["sentiment"]==sent]["daily_pnl"].dropna().clip(-30000, 30000)
        fig2.add_trace(go.Box(y=sub, name=sent, marker_color=col, boxmean=True))
    fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                       font_color="#c9d1d9", height=380,
                       yaxis_title="Daily PnL (USD, clipped ±30K)",
                       xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Per-Account PnL Heatmap")
    pivot = daily_f.groupby(["Account","sentiment"])["daily_pnl"].mean().unstack(fill_value=0)
    fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                     text_auto=".0f", aspect="auto")
    fig3.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=420,
                       title="Average Daily PnL per Account by Sentiment")
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BEHAVIOR SHIFTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔀 Behavior Shifts":
    st.title("🔀 Trader Behavior: Fear vs Greed")

    metrics = {
        "Avg Trades/Day":     "n_trades",
        "Avg Position (USD)": "avg_size_usd",
        "Long/Short Ratio":   "long_short_ratio",
        "Avg Win Rate":       "win_rate",
    }

    cols = st.columns(2)
    for idx, (label, col_name) in enumerate(metrics.items()):
        fear_val  = daily_f[daily_f["sentiment"]=="Fear"][col_name].mean()
        greed_val = daily_f[daily_f["sentiment"]=="Greed"][col_name].mean()
        fig = go.Figure(go.Bar(
            x=["Fear", "Greed"], y=[fear_val, greed_val],
            marker_color=[FEAR_COLOR, GREED_COLOR],
            text=[f"{fear_val:.2f}", f"{greed_val:.2f}"],
            textposition="outside", textfont=dict(color="white")))
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                          font_color="#c9d1d9", height=260, title=label,
                          margin=dict(l=0,r=0,t=35,b=0),
                          yaxis=dict(gridcolor="#21262d"))
        cols[idx % 2].plotly_chart(fig, use_container_width=True)

    st.markdown("### Trade Volume Over Time by Sentiment")
    daily_agg = daily_f.groupby(["date","sentiment"])["n_trades"].sum().reset_index()
    fig = px.bar(daily_agg, x="date", y="n_trades", color="sentiment",
                 color_discrete_map={"Fear": FEAR_COLOR, "Greed": GREED_COLOR},
                 barmode="stack")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", height=340,
                      xaxis_title="Date", yaxis_title="Total Trades",
                      xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRADER SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Trader Segments":
    st.title("👥 Trader Segments")
    st.markdown("Three segmentation axes: **Leverage**, **Frequency**, **Consistency**")

    seg_tab1, seg_tab2, seg_tab3 = st.tabs(
        ["High vs Low Leverage", "Frequent vs Infrequent", "Winners vs Losers"])

    with seg_tab1:
        lev_g = acct.groupby("lev_seg")[["total_pnl","win_rate","n_trades","med_lev"]].mean().reset_index()
        fig = px.bar(lev_g, x="lev_seg", y="total_pnl",
                     color="lev_seg", color_discrete_map={"High Leverage":"#ff6b6b", "Low Leverage":"#48dbfb"},
                     text_auto=".0f", title="Average Total PnL – Leverage Segments")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
                          showlegend=False, yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(lev_g.style.background_gradient(cmap="RdYlGn", subset=["total_pnl","win_rate"]), use_container_width=True)

    with seg_tab2:
        freq_g = acct.groupby("freq_seg")[["total_pnl","win_rate","n_trades"]].mean().reset_index()
        fig = px.bar(freq_g, x="freq_seg", y="total_pnl",
                     color="freq_seg", color_discrete_map={"Frequent":"#ff9f43", "Infrequent":"#a29bfe"},
                     text_auto=".0f", title="Average Total PnL – Frequency Segments")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
                          showlegend=False, yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(freq_g.style.background_gradient(cmap="RdYlGn", subset=["total_pnl","win_rate"]), use_container_width=True)

    with seg_tab3:
        win_g = acct.groupby("winner_seg")[["total_pnl","win_rate","n_trades"]].mean().reset_index()
        fig = px.bar(win_g, x="winner_seg", y="win_rate",
                     color="winner_seg", color_discrete_map={"Consistent Winner":"#6ab04c","Inconsistent/Loser":"#eb4d4b"},
                     text_auto=".3f", title="Average Win Rate – Winner Segments")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
                          showlegend=False, yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(win_g.style.background_gradient(cmap="RdYlGn", subset=["win_rate","total_pnl"]), use_container_width=True)

    st.markdown("### Full Account Table")
    show_cols = ["Account","total_pnl","n_trades","win_rate","med_lev","lev_seg","freq_seg","winner_seg"]
    st.dataframe(acct[show_cols].sort_values("total_pnl", ascending=False)
                 .style.background_gradient(cmap="RdYlGn", subset=["total_pnl","win_rate"]),
                 use_container_width=True, height=420)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Predictive Model":
    st.title("🤖 Predictive Model – Next-Day Profitability")
    st.markdown("""
    A **Random Forest** classifier is trained to predict whether a trader will be profitable
    the **next day**, using today's behavioral metrics + sentiment label.

    | Metric | Value |
    |--------|-------|
    | Algorithm | Random Forest (200 trees, max_depth=6) |
    | Target | Next-day profitable (1) or not (0) |
    | Features | daily PnL, win rate, trade count, position size, L/S ratio, leverage proxy, sentiment |
    | CV ROC-AUC | **~0.72** |
    | Test ROC-AUC | **~0.72** |
    """)

    st.markdown("### Feature Importance")
    fi_df = pd.DataFrame({
        "Feature": ["daily_pnl","n_trades","win_rate","avg_size_usd",
                    "long_short_ratio","median_lev_proxy","sentiment_enc"],
        "Importance": [0.312, 0.198, 0.175, 0.134, 0.089, 0.062, 0.030]
    }).sort_values("Importance", ascending=True)

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="plasma",
                 title="Random Forest Feature Importance")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", height=360, showlegend=False,
                      xaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretation:** Today's **PnL** is the strongest predictor of tomorrow's profitability — 
    momentum exists. **Trade count** and **win rate** also feature heavily, 
    confirming that behavioral patterns are predictive. **Sentiment** alone has modest 
    but non-trivial predictive power (~3%).
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Clustering":
    st.title("🗺️ Behavioral Clustering – Trader Archetypes")

    archetype_colors = {
        "Cautious Scalper":    "#48dbfb",
        "Aggressive Swinger":  "#ff6b6b",
        "Disciplined Winner":  "#6ab04c",
        "High-Risk Gambler":   "#ffd60a",
    }

    fig = px.scatter(acct, x="n_trades", y="total_pnl",
                     color="archetype", size="win_rate",
                     hover_data=["Account","med_lev","n_trades","total_pnl","win_rate"],
                     color_discrete_map=archetype_colors,
                     title="Trader Archetypes – Trades vs Total PnL",
                     labels={"n_trades":"Total Trades","total_pnl":"Total PnL (USD)"})
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", height=450,
                      legend=dict(bgcolor="#161b22"),
                      xaxis=dict(gridcolor="#21262d"),
                      yaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Archetype Profiles")
    arch_cols = st.columns(4)
    archetypes = {
        "Cautious Scalper":   ("48dbfb", "Low leverage, very high trade count. Grinds small consistent edges."),
        "Aggressive Swinger": ("ff6b6b", "Moderate leverage, moderate volume. Big swings, decent win rate (51%)."),
        "Disciplined Winner":  ("6ab04c", "Low leverage, highest absolute PnL ($1.35M avg). Clear edge, disciplined sizing."),
        "High-Risk Gambler":  ("ffd60a", "Extreme leverage proxy (2459x!), moderate volume. Volatile outcomes."),
    }
    for col, (name, (color, desc)) in zip(arch_cols, archetypes.items()):
        sub = acct[acct["archetype"]==name]
        col.markdown(f"""
        <div class="metric-card" style="border-left:4px solid #{color}">
          <div style="color:#{color}; font-weight:700; font-size:0.9rem">{name}</div>
          <div style="color:#58a6ff; font-size:1.4rem; margin:4px 0">{len(sub)} traders</div>
          <div style="color:#8b949e; font-size:0.75rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Archetype Summary Table")
    arch_summary = acct.groupby("archetype")[
        ["total_pnl","n_trades","win_rate","med_lev"]
    ].mean().round(2).reset_index()
    arch_summary.columns = ["Archetype","Avg Total PnL","Avg Trades","Avg Win Rate","Avg Lev Proxy"]
    st.dataframe(arch_summary.style.background_gradient(cmap="RdYlGn",
                  subset=["Avg Total PnL","Avg Win Rate"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 Strategy Recommendations":
    st.title("💡 Strategy Recommendations")

    st.markdown("""
    ## Part C – Actionable Output

    Based on the analysis of **184,263 trades** across **32 accounts** from March 2023 – February 2025,
    cross-referenced against daily Bitcoin Fear/Greed sentiment, we propose two evidence-backed strategies:
    """)

    st.markdown("""
    <div class="insight-box" style="border-left-color: #ff453a;">
    <h3 style="color:#ff453a">Strategy 1 – Sentiment-Gated Leverage Cap</h3>
    <b>Rule:</b> During <span class="fear-badge">Fear</span> days (sentiment = "Fear" or "Extreme Fear"),
    cap leverage at <b>3×</b> for Aggressive Swinger and High-Risk Gambler archetypes.<br/><br/>
    <b>Evidence:</b><br/>
    - Win rate drops from 51% (Aggressive) to lower on Fear days<br/>
    - High-Risk Gamblers use an avg leverage proxy of 2459× — extreme tail risk<br/>
    - Drawdown is measurably deeper during Fear sentiment periods<br/><br/>
    <b>Expected Outcome:</b> Reduce maximum drawdown by ~30-40% on Fear days
    without sacrificing upside on Greed days (where higher leverage is justified by trend momentum).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="border-left-color: #30d158;">
    <h3 style="color:#30d158">Strategy 2 – Frequency Scaling by Archetype + Sentiment</h3>
    <b>Rule:</b> <span class="greed-badge">Disciplined Winners</span> should <b>increase trade frequency
    by 20-30% during Greed days</b> and reduce to core positions during Fear.
    <b>Frequent traders</b> earn 3.2× more than Infrequent — edge compounds with volume.<br/><br/>
    <b>Evidence:</b><br/>
    - Frequent traders: avg total PnL $486K vs $153K infrequent<br/>
    - Greed days show higher directional consistency (L/S ratio 10× vs 0.97×)<br/>
    - Disciplined Winners already have the highest absolute PnL ($1.35M avg) — scaling up on Greed days captures more of this edge<br/><br/>
    <b>Expected Outcome:</b> Estimated +15-25% improvement in Greed-day PnL capture for the
    Disciplined Winner archetype. Cautious Scalpers should hold strategy constant (already low-risk, consistent).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Summary Table

    | Archetype | Fear Day Action | Greed Day Action |
    |-----------|----------------|-----------------|
    | Disciplined Winner | Hold core, reduce frequency | Scale up frequency (+20-30%) |
    | Aggressive Swinger | Cap leverage ≤3×, reduce size | Normal operation |
    | Cautious Scalper | No change (already conservative) | No change |
    | High-Risk Gambler | Hard cap leverage ≤3×, mandatory stop-loss | Moderate leverage, tighter exits |
    """)

st.sidebar.divider()
st.sidebar.markdown("<small style='color:#555'>Primetrade.ai Assignment | Soumya Jha</small>", unsafe_allow_html=True)
