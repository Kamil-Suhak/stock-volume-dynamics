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

    g['spread'] = (g['high'] - g['low']) / g['open']
    g['abs_return'] = ((g['close'] - g['open']) / g['open']).abs()
    g['overnight_gap'] = ((g['open'] - g['close'].shift(1)) / g['close'].shift(1)).abs()

    # Lags for forward prediction
    g['lag_spread'] = g['spread'].shift(1)
    g['lag_abs_return'] = g['abs_return'].shift(1)
    trade_size = g['volume'] / g['transactions'].replace(0, 1)
    g['lag_log_trade_size'] = np.log(trade_size.clip(lower=1.0)).shift(1)

    g['lag_vol_1h'] = g['target_log_vol'].shift(1)
    g['lag_vol_2h'] = g['target_log_vol'].shift(2)
    g['vol_momentum'] = g['lag_vol_1h'] - g['lag_vol_2h']
    g['lag_vol_same_hour'] = g['target_log_vol'].shift(7)

    # 5-hour rolling spread & 20-day range position (0 = low, 1 = high)
    g['rolling_spread_5h'] = g['spread'].rolling(5, min_periods=1).mean().shift(1)
    rolling_high_20d = g['high'].rolling(140, min_periods=20).max().shift(1)
    rolling_low_20d = g['low'].rolling(140, min_periods=20).min().shift(1)
    g['range_pos_20d'] = (g['close'].shift(1) - rolling_low_20d) / (rolling_high_20d - rolling_low_20d + 1e-8)

    return g


def build_features(df, mode="forward"):
    if df.empty:
        raise ValueError("DataFrame is empty.")

    processed = [calc_ticker_features(g) for _, g in df.groupby('ticker')]
    df_feat = pd.concat(processed, ignore_index=True)
    df_feat = df_feat.sort_values(['timestamp', 'ticker']).reset_index(drop=True)

    df_clean = df_feat.dropna().reset_index(drop=True)

    # Dummies for hour, day, and ticker
    hour_dummies = pd.get_dummies(df_clean['hour'], prefix='hour', drop_first=True, dtype=float)
    day_dummies = pd.get_dummies(df_clean['day_of_week'], prefix='day', drop_first=True, dtype=float)
    dummy_dfs = [hour_dummies, day_dummies]

    if df_clean['ticker'].nunique() > 1:
        ticker_dummies = pd.get_dummies(df_clean['ticker'], prefix='ticker', drop_first=True, dtype=float)
        dummy_dfs.append(ticker_dummies)

    calendar_df = pd.concat(dummy_dfs, axis=1)

    fwd_cols = [
        'lag_spread', 'lag_abs_return', 'overnight_gap', 'lag_log_trade_size',
        'lag_vol_1h', 'lag_vol_2h', 'vol_momentum', 'lag_vol_same_hour',
        'rolling_spread_5h', 'range_pos_20d', 'log_news_surge'
    ]
    exp_cols = [
        'spread', 'abs_return', 'overnight_gap', 'lag_log_trade_size',
        'lag_vol_1h', 'lag_vol_2h', 'vol_momentum', 'lag_vol_same_hour',
        'rolling_spread_5h', 'range_pos_20d', 'log_news_surge'
    ]

    X_forward = pd.concat([df_clean[fwd_cols], calendar_df], axis=1)
    X_explanatory = pd.concat([df_clean[exp_cols], calendar_df], axis=1)
    y = df_clean['target_log_vol']

    if mode == "forward":
        return X_forward, y, df_clean
    elif mode == "explanatory":
        return X_explanatory, y, df_clean
    elif mode == "both":
        return X_forward, X_explanatory, y, df_clean
    else:
        raise ValueError(f"Unknown mode '{mode}'")
