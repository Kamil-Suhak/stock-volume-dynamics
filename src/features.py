import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db import get_connection


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
    con.close()

    if bars_df.empty:
        return bars_df

    bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'], utc=True)
    bars_df['timestamp_et'] = bars_df['timestamp'].dt.tz_convert('America/New_York')
    bars_df['hour'] = bars_df['timestamp_et'].dt.hour
    bars_df['day_of_week'] = bars_df['timestamp_et'].dt.day_name()

    if rth_only:
        bars_df = bars_df[(bars_df['hour'] >= 9) & (bars_df['hour'] <= 15)].copy()

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
    g['lag_rvol_same_hour'] = (g['target_log_vol'] - g['base_intraday']).shift(7)

    # Market price dynamics
    g['spread'] = (g['high'] - g['low']) / g['open']
    g['abs_return'] = ((g['close'] - g['open']) / g['open']).abs()
    g['overnight_gap'] = ((g['open'] - g['close'].shift(1)) / g['close'].shift(1)).abs()

    g['lag_spread'] = g['spread'].shift(1)
    g['lag_abs_return'] = g['abs_return'].shift(1)

    # Fundamental turnover anchor
    shares = g['shares_outstanding'].iloc[0] if 'shares_outstanding' in g and pd.notnull(g['shares_outstanding'].iloc[0]) else 1e9
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
    dummy_dfs = [hour_dummies]

    if df_clean['ticker'].nunique() > 1:
        ticker_dummies = pd.get_dummies(df_clean['ticker'], prefix='ticker', drop_first=True, dtype=float)
        dummy_dfs.append(ticker_dummies)

    calendar_df = pd.concat(dummy_dfs, axis=1)

    fwd_cols = [
        'regime_ratio', 'lag_rvol_1h', 'lag_rvol_2h', 'lag_rvol_same_hour',
        'lag_spread', 'lag_abs_return', 'overnight_gap', 'lag_turnover'
    ]
    exp_cols = [
        'regime_ratio', 'lag_rvol_1h', 'lag_rvol_same_hour',
        'spread', 'abs_return', 'overnight_gap', 'lag_turnover'
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
