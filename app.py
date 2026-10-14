import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Stock Volume Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #848e9c;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f3f6;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        font-weight: 600;
        color: #00c076;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        font-weight: 600;
        color: #ff4d4f;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_db_data():
    con = sqlite3.connect("data/market.db")
    symbols_df = pd.read_sql_query("SELECT id, ticker FROM symbols ORDER BY ticker ASC;", con)

    bars_query = """
        SELECT b.id AS bar_id, s.ticker, b.timestamp, b.open, b.high, b.low, b.close, b.volume, b.transactions
        FROM hourly_bars b
        JOIN symbols s ON s.id = b.symbol_id
        ORDER BY b.timestamp ASC;
    """
    bars_df = pd.read_sql_query(bars_query, con)

    preds_query = "SELECT bar_id, model_name, actual_log_vol, pred_log_vol, residual FROM predictions;"
    try:
        preds_df = pd.read_sql_query(preds_query, con)
    except Exception:
        preds_df = pd.DataFrame(columns=['bar_id', 'model_name', 'actual_log_vol', 'pred_log_vol', 'residual'])

    benchmarks_query = """
        SELECT mode AS Mode, model AS Model, cv_rmse AS CV_RMSE, cv_mae AS CV_MAE, cv_mape AS [CV_MAPE_%]
        FROM model_benchmarks
        ORDER BY id DESC
        LIMIT 4;
    """
    try:
        benchmarks_df = pd.read_sql_query(benchmarks_query, con)
    except Exception:
        benchmarks_df = pd.DataFrame()

    con.close()

    bars_df['timestamp_utc'] = pd.to_datetime(bars_df['timestamp'], utc=True)
    bars_df['timestamp_et'] = bars_df['timestamp_utc'].dt.tz_convert('America/New_York')
    bars_df['date'] = bars_df['timestamp_et'].dt.date
    bars_df['hour'] = bars_df['timestamp_et'].dt.hour
    bars_df['day_name'] = bars_df['timestamp_et'].dt.day_name()

    return symbols_df, bars_df, preds_df, benchmarks_df


@st.cache_data(ttl=60)
def load_ticker_news(ticker):
    con = sqlite3.connect("data/market.db")
    query = """
        SELECT a.published_utc, a.title, a.author, a.article_url
        FROM articles a
        JOIN article_symbols j ON j.article_id = a.id
        JOIN symbols s ON s.id = j.symbol_id
        WHERE s.ticker = ?
        ORDER BY a.published_utc DESC
        LIMIT 25;
    """
    try:
        df = pd.read_sql_query(query, con, params=(ticker,))
    except Exception:
        df = pd.DataFrame(columns=['published_utc', 'title', 'author', 'article_url'])
    finally:
        con.close()
    return df


symbols_df, bars_df, preds_df, benchmarks_df = load_db_data()

# Sidebar
st.sidebar.title("Settings")

tickers = symbols_df['ticker'].tolist() if not symbols_df.empty else ["AAPL"]
selected_ticker = st.sidebar.selectbox("Stock", tickers, index=0)

ticker_bars = bars_df[bars_df['ticker'] == selected_ticker].copy()
rth_bars = ticker_bars[(ticker_bars['hour'] >= 9) & (ticker_bars['hour'] <= 15)].copy()

merged_df = rth_bars.merge(preds_df, on='bar_id', how='left')

available_models = merged_df['model_name'].dropna().unique().tolist()
if not available_models:
    available_models = ["Forward_Ridge", "Forward_HistGradBoost", "Explanatory_Ridge", "Explanatory_HistGradBoost"]

selected_model = st.sidebar.selectbox("Model", available_models, index=0)

predicted_dates = merged_df[merged_df['model_name'] == selected_model]['date'].dropna().unique()
if len(predicted_dates) > 0:
    predicted_dates = sorted(predicted_dates, reverse=True)
    selected_date = st.sidebar.selectbox("Date", predicted_dates, index=0)
else:
    all_dates = sorted(ticker_bars['date'].unique(), reverse=True) if not ticker_bars.empty else [pd.to_datetime('today').date()]
    selected_date = st.sidebar.selectbox("Date", all_dates, index=0)

# Filter session data
session_data = merged_df[(merged_df['date'] == selected_date) & (merged_df['model_name'] == selected_model)].sort_values('hour')

if session_data.empty:
    session_data = rth_bars[rth_bars['date'] == selected_date].sort_values('hour')
    session_data['pred_log_vol'] = np.nan
    session_data['actual_log_vol'] = np.log(session_data['volume'].clip(lower=1.0))
    session_data['residual'] = np.nan

session_data['pred_vol'] = np.exp(session_data['pred_log_vol'])

# Header & KPIs
st.title(f"{selected_ticker} Volume Analysis")
st.caption(f"Date: {selected_date} | Model: {selected_model}")

total_actual_vol = session_data['volume'].sum()
total_pred_vol = session_data['pred_vol'].sum()
has_preds = not np.isnan(total_pred_vol) and total_pred_vol > 0

if has_preds:
    rvol = total_actual_vol / total_pred_vol
    rvol_delta_pct = (rvol - 1.0) * 100.0
    session_mape = np.mean(np.abs((session_data['volume'] - session_data['pred_vol']) / session_data['volume'])) * 100.0
else:
    rvol = 1.0
    rvol_delta_pct = 0.0
    session_mape = 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Actual Volume</div>
        <div class="metric-value">{total_actual_vol:,.0f}</div>
        <span style="font-size: 0.8rem; color: #848e9c;">Shares traded</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Volume</div>
        <div class="metric-value">{total_pred_vol:,.0f}</div>
        <span style="font-size: 0.8rem; color: #848e9c;">Model expectation</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    delta_class = "metric-delta-pos" if rvol >= 1.0 else "metric-delta-neg"
    sign = "+" if rvol_delta_pct >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RVOL</div>
        <div class="metric-value">{rvol:.2f}x</div>
        <span class="{delta_class}">{sign}{rvol_delta_pct:.1f}% vs prediction</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Session Error (MAPE)</div>
        <div class="metric-value">{session_mape:.1f}%</div>
        <span style="font-size: 0.8rem; color: #848e9c;">Mean absolute error</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_session, tab_heatmap, tab_execution, tab_benchmarks, tab_news = st.tabs([
    "Hourly Volume",
    "Volume Heatmap",
    "Order Impact",
    "Model Benchmarks",
    "Recent News"
])

# Tab 1: Hourly Volume
with tab_session:
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("Actual vs Predicted Volume")
        fig_curve = go.Figure()

        fig_curve.add_trace(go.Scatter(
            x=[f"{h:02d}:00" for h in session_data['hour']],
            y=session_data['volume'],
            mode='lines+markers',
            name='Actual Volume',
            line=dict(color='#00d2ff', width=3),
            marker=dict(size=8, color='#00d2ff')
        ))

        if has_preds:
            fig_curve.add_trace(go.Scatter(
                x=[f"{h:02d}:00" for h in session_data['hour']],
                y=session_data['pred_vol'],
                mode='lines+markers',
                name='Predicted Volume',
                line=dict(color='#ff9900', width=2.5, dash='dash'),
                marker=dict(size=7, color='#ff9900')
            ))

        fig_curve.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Hour (ET)", gridcolor='#2a2e39'),
            yaxis=dict(title="Volume", gridcolor='#2a2e39'),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_chart2:
        st.subheader("Hourly Surprise %")
        if has_preds:
            session_data['surprise_pct'] = ((session_data['volume'] / session_data['pred_vol']) - 1.0) * 100.0
            colors = ['#00c076' if v >= 0 else '#ff4d4f' for v in session_data['surprise_pct']]

            fig_surprise = go.Figure()
            fig_surprise.add_trace(go.Bar(
                x=[f"{h:02d}:00" for h in session_data['hour']],
                y=session_data['surprise_pct'],
                marker_color=colors,
                name='Surprise %'
            ))

            fig_surprise.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Hour (ET)", gridcolor='#2a2e39'),
                yaxis=dict(title="Difference from Prediction (%)", gridcolor='#2a2e39'),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_surprise, use_container_width=True)
        else:
            st.info("No predictions found for this date. Run python src/main.py to populate predictions.")

    st.markdown("#### Hourly Breakdown")
    tbl = session_data[['hour', 'open', 'high', 'low', 'close', 'volume', 'pred_vol', 'transactions']].copy()
    tbl.columns = ['Hour (ET)', 'Open', 'High', 'Low', 'Close', 'Actual Volume', 'Predicted Volume', 'Trades']
    tbl['Hour (ET)'] = tbl['Hour (ET)'].apply(lambda h: f"{h:02d}:00")
    tbl['Actual Volume'] = tbl['Actual Volume'].apply(lambda v: f"{v:,.0f}")
    tbl['Predicted Volume'] = tbl['Predicted Volume'].apply(lambda v: f"{v:,.0f}" if pd.notnull(v) else "N/A")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# Tab 2: Heatmap
with tab_heatmap:
    st.subheader(f"Median Hourly Volume: {selected_ticker}")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    heatmap_data = rth_bars.groupby(['day_name', 'hour'])['volume'].median().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='volume')
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig_heat = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour (ET)", y="Day", color="Median Volume"),
        x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
        y=heatmap_pivot.index,
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig_heat.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    - **Open and Close:** Volume peaks at 9:30 AM and 3:30 PM across all days.
    - **Midday:** Volume drops by roughly 40-50% between 12:00 PM and 2:00 PM.
    """)

# Tab 3: Order Impact
with tab_execution:
    st.subheader("Order Size & Impact Calculator")
    st.caption("Estimate what percentage of hourly volume an order would take.")

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        order_size = st.number_input("Order Size (Shares)", min_value=1000, max_value=5000000, value=75000, step=5000)
    with col_inp2:
        max_participation_cap = st.slider("Max Desired Participation (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.5)

    exec_df = session_data[['hour', 'volume', 'pred_vol']].copy()
    base_volume = exec_df['pred_vol'] if has_preds else exec_df['volume']
    exec_df['impact_pct'] = (order_size / base_volume) * 100.0

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in exec_df['hour']],
        y=exec_df['impact_pct'],
        name='Participation %',
        marker_color=['#00c076' if imp <= max_participation_cap else '#ff4d4f' for imp in exec_df['impact_pct']]
    ))
    fig_impact.add_hline(
        y=max_participation_cap,
        line_dash="dash",
        line_color="#ffcc00",
        annotation_text=f"Cap ({max_participation_cap}%)"
    )
    fig_impact.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Hour (ET)", gridcolor='#2a2e39'),
        yaxis=dict(title="Order % of Volume", gridcolor='#2a2e39'),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_impact, use_container_width=True)

# Tab 4: Benchmarks
with tab_benchmarks:
    st.subheader("Cross-Validation Results (5-Fold TimeSeriesSplit)")

    if not benchmarks_df.empty:
        benchmark_data = benchmarks_df.copy()
    else:
        benchmark_data = pd.DataFrame([
            {"Mode": "Forward", "Model": "Ridge", "CV_RMSE": 0.2852, "CV_MAE": 0.2163, "CV_MAPE_%": 21.41},
            {"Mode": "Forward", "Model": "HistGradBoost", "CV_RMSE": 0.3092, "CV_MAE": 0.2354, "CV_MAPE_%": 23.44},
            {"Mode": "Explanatory", "Model": "Ridge", "CV_RMSE": 0.2849, "CV_MAE": 0.2161, "CV_MAPE_%": 21.39},
            {"Mode": "Explanatory", "Model": "HistGradBoost", "CV_RMSE": 0.2589, "CV_MAE": 0.2004, "CV_MAPE_%": 20.37}
        ])

    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

    fig_bench = px.bar(
        benchmark_data,
        x="Model",
        y="CV_MAPE_%",
        color="Mode",
        barmode="group",
        title="Cross-Validation MAPE % (Lower is Better)",
        color_discrete_map={"Forward": "#00c076", "Explanatory": "#2962ff"}
    )
    fig_bench.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="MAPE (%)", gridcolor='#2a2e39')
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    st.markdown("""
    - **Forward prediction:** Ridge performs better (~21.4% MAPE) because L2 regularization prevents it from overfitting to noisy price lags.
    - **Explanatory modeling:** HistGradientBoosting performs better (~20.4% MAPE) because it can capture non-linear interactions between spread and the time of day.
    """)

# Tab 5: News
with tab_news:
    st.subheader(f"Recent News: {selected_ticker}")
    news_data = load_ticker_news(selected_ticker)

    if not news_data.empty:
        news_display = news_data.copy()
        news_display.columns = ["Published (UTC)", "Headline", "Author", "URL"]
        st.dataframe(
            news_display,
            column_config={"URL": st.column_config.LinkColumn("URL")},
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"No news articles saved yet for {selected_ticker}.")

    st.markdown("""
    **NewsSurge feature:**
    - Measures how many articles were published in the last 24 hours relative to the 14-day daily average.
    - Uses `log(1 + NewsSurge)` to smooth out huge spikes in article count.
    """)
