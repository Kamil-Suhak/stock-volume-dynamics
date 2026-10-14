import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db import get_connection


def compute_news_surge(con, tickers=None):
    query = """
        SELECT s.ticker, a.published_utc
        FROM articles a
        JOIN article_symbols j ON j.article_id = a.id
        JOIN symbols s ON s.id = j.symbol_id
    """
    params = ()
    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]
        placeholders = ",".join(["?"] * len(tickers))
        query += f" WHERE s.ticker IN ({placeholders})"
        params = tuple(tickers)

    query += " ORDER BY a.published_utc ASC;"

    try:
        news_df = pd.read_sql_query(query, con, params=params)
    except Exception:
        return pd.DataFrame(columns=['ticker', 'timestamp_round', 'log_news_surge'])

    if news_df.empty:
        return pd.DataFrame(columns=['ticker', 'timestamp_round', 'log_news_surge'])

    news_df['published_utc'] = pd.to_datetime(news_df['published_utc'], utc=True)
    news_df['timestamp_round'] = news_df['published_utc'].dt.floor('h')

    records = []
    for ticker, group in news_df.groupby('ticker'):
        hourly = group.set_index('timestamp_round').resample('1h').size()
        extended_end = hourly.index.max() + pd.Timedelta(hours=24)
        extended_idx = pd.date_range(start=hourly.index.min(), end=extended_end, freq='1h', tz='UTC')
        hourly = hourly.reindex(extended_idx, fill_value=0)

        rolling_24h = hourly.rolling(24, min_periods=1).sum()
        baseline_14d = rolling_24h.shift(1).rolling(14 * 24, min_periods=24).mean().fillna(0)

        surge = rolling_24h / (baseline_14d + 1.0)
        log_surge = np.log1p(surge)

        s_df = pd.DataFrame({
            'ticker': ticker,
            'timestamp_round': hourly.index,
            'log_news_surge': log_surge.values
        })
        records.append(s_df)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=['ticker', 'timestamp_round', 'log_news_surge'])


def load_data(tickers=None, rth_only=True):
    con = get_connection()
    query = """
        SELECT 
            s.ticker,
            s.shares_outstanding,
            s.market_cap,
            b.id AS bar_id,
            b.timestamp,
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.transactions
        FROM hourly_bars b
        JOIN symbols s ON s.id = b.symbol_id
    """
    params = ()
    if tickers:
        if isinstance(tickers, str):
            tickers = [tickers]
        placeholders = ",".join(["?"] * len(tickers))
        query += f" WHERE s.ticker IN ({placeholders})"
        params = tuple(tickers)

    query += " ORDER BY s.ticker, b.timestamp ASC;"

    bars_df = pd.read_sql_query(query, con, params=params)
    news_surge_df = compute_news_surge(con, tickers)
    con.close()

    if bars_df.empty:
        return bars_df

    bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'], utc=True)
    bars_df['timestamp_et'] = bars_df['timestamp'].dt.tz_convert('America/New_York')
    bars_df['hour'] = bars_df['timestamp_et'].dt.hour
    bars_df['day_of_week'] = bars_df['timestamp_et'].dt.day_name()

    half_day_dates = {'2024-11-29', '2024-12-24', '2025-07-03', '2025-11-28', '2025-12-24', '2026-07-03'}
    bars_df['is_half_day'] = bars_df['timestamp_et'].dt.strftime('%Y-%m-%d').isin(half_day_dates).astype(float)

    if rth_only:
        bars_df = bars_df[(bars_df['hour'] >= 9) & (bars_df['hour'] <= 15)].copy()

    if not news_surge_df.empty:
        bars_df['timestamp_round'] = bars_df['timestamp'].dt.floor('h')
        bars_df = bars_df.merge(news_surge_df, on=['ticker', 'timestamp_round'], how='left')
        bars_df['log_news_surge'] = bars_df['log_news_surge'].fillna(0.0)
        bars_df.drop(columns=['timestamp_round'], inplace=True, errors='ignore')
    else:
        bars_df['log_news_surge'] = 0.0

    return bars_df.reset_index(drop=True)


def calc_ticker_features(group):
    g = group.sort_values('timestamp').copy()
    g['target_log_vol'] = np.log(g['volume'].clip(lower=1.0))

    # Rolling intraday baselines (shift 1 per hour to avoid lookahead)
    g['base_intraday'] = g.groupby('hour')['target_log_vol'].transform(
        lambda s: s.shift(1).rolling(15, min_periods=3).median()
    ).bfill()

    g['base_5d'] = g.groupby('hour')['target_log_vol'].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).median()
    ).bfill()

    g['regime_ratio'] = g['base_5d'] - g['base_intraday']
    g['target_dev'] = g['target_log_vol'] - g['base_intraday']

    # Relative volume (RVOL) lags
    g['lag_rvol_1h'] = (g['target_log_vol'] - g['base_intraday']).shift(1)
    g['lag_rvol_2h'] = (g['target_log_vol'] - g['base_intraday']).shift(2)
    g['rvol_momentum'] = g['lag_rvol_1h'] - g['lag_rvol_2h']
    g['lag_rvol_same_hour'] = (g['target_log_vol'] - g['base_intraday']).shift(7)

    # Market price dynamics
    g['spread'] = (g['high'] - g['low']) / g['open']
    g['abs_return'] = ((g['close'] - g['open']) / g['open']).abs()
    g['overnight_gap'] = ((g['open'] - g['close'].shift(1)) / g['close'].shift(1)).abs()
    g['gap_shock'] = (g['overnight_gap'] > 0.025).astype(float)

    trade_size = g['volume'] / g['transactions'].replace(0, 1)
    g['lag_log_trade_size'] = np.log(trade_size.clip(lower=1.0)).shift(1)

    g['lag_spread'] = g['spread'].shift(1)
    g['lag_abs_return'] = g['abs_return'].shift(1)
    g['rolling_spread_5h'] = g['spread'].rolling(5, min_periods=1).mean().shift(1)
    g['vol_std_20h'] = g['target_log_vol'].rolling(20, min_periods=5).std().shift(1).fillna(0.3)

    # Fundamental float & turnover
    shares = g['shares_outstanding'].iloc[0] if 'shares_outstanding' in g and pd.notnull(g['shares_outstanding'].iloc[0]) else 1e9
    mcap = g['market_cap'].iloc[0] if 'market_cap' in g and pd.notnull(g['market_cap'].iloc[0]) else 1e11
    g['log_mcap'] = float(np.log(max(mcap, 1.0)))
    g['lag_turnover'] = np.log(g['volume'].shift(1) / max(shares, 1.0)).fillna(-10.0)

    return g


def build_features(df, mode="forward"):
    if df.empty:
        raise ValueError("DataFrame is empty.")

    processed = [calc_ticker_features(g) for _, g in df.groupby('ticker')]
    df_feat = pd.concat(processed, ignore_index=True)
    df_feat = df_feat.sort_values(['timestamp', 'ticker']).reset_index(drop=True)

    df_clean = df_feat.dropna().reset_index(drop=True)

    hour_dummies = pd.get_dummies(df_clean['hour'], prefix='hour', drop_first=True, dtype=float)
    day_dummies = pd.get_dummies(df_clean['day_of_week'], prefix='day', drop_first=True, dtype=float)
    dummy_dfs = [hour_dummies, day_dummies]

    if df_clean['ticker'].nunique() > 1:
        ticker_dummies = pd.get_dummies(df_clean['ticker'], prefix='ticker', drop_first=True, dtype=float)
        dummy_dfs.append(ticker_dummies)

    calendar_df = pd.concat(dummy_dfs, axis=1)

    fwd_cols = [
        'regime_ratio', 'lag_rvol_1h', 'lag_rvol_2h', 'rvol_momentum', 'lag_rvol_same_hour',
        'lag_spread', 'lag_abs_return', 'overnight_gap', 'gap_shock', 'lag_log_trade_size',
        'rolling_spread_5h', 'vol_std_20h', 'lag_turnover', 'log_mcap', 'is_half_day', 'log_news_surge'
    ]
    exp_cols = [
        'regime_ratio', 'lag_rvol_1h', 'rvol_momentum', 'lag_rvol_same_hour',
        'spread', 'abs_return', 'overnight_gap', 'gap_shock', 'lag_log_trade_size',
        'rolling_spread_5h', 'vol_std_20h', 'lag_turnover', 'log_mcap', 'is_half_day', 'log_news_surge'
    ]

    X_forward = pd.concat([df_clean[fwd_cols], calendar_df], axis=1)
    X_explanatory = pd.concat([df_clean[exp_cols], calendar_df], axis=1)
    y = df_clean['target_dev']

    if mode == "forward":
        return X_forward, y, df_clean
    elif mode == "explanatory":
        return X_explanatory, y, df_clean
    elif mode == "both":
        return X_forward, X_explanatory, y, df_clean
    else:
        raise ValueError(f"Unknown mode '{mode}'")
