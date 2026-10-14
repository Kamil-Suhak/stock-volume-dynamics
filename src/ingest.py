import json
import sys
import time
import requests
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db import init_db, get_connection, get_or_create_symbol, insert_hourly_bars, get_latest_timestamp, update_symbol_fundamentals

BASE_URL = "https://api.massive.com/v2/aggs/ticker/"
API_KEY = "BJox5qIIzSxo2VSBknxLN7d8uQYs7uge"
RAW_DATA_DIR = BASE_DIR / "raw_data"
SLEEP_TIME = 12.5
_LAST_REQUEST_TIME = 0.0


def get_next_chunk_index(ticker_dir, prefix):
    existing = []
    for f in ticker_dir.glob(f"*{prefix}*.json"):
        for part in f.stem.split("_"):
            if part.isdigit():
                existing.append(int(part))
    return (max(existing) + 1) if existing else 1


def request_with_retry(url, params=None, max_retries=5):
    global _LAST_REQUEST_TIME
    for attempt in range(max_retries):
        elapsed = time.time() - _LAST_REQUEST_TIME
        if _LAST_REQUEST_TIME > 0 and elapsed < SLEEP_TIME:
            time.sleep(SLEEP_TIME - elapsed)
        try:
            _LAST_REQUEST_TIME = time.time()
            res = requests.get(url, params=params, timeout=30)
            if res.status_code == 429:
                print("Rate limit reached. Sleeping 30s...")
                time.sleep(30)
                _LAST_REQUEST_TIME = time.time()
                continue
            if res.status_code in (500, 502, 503, 504):
                print(f"Server error {res.status_code}. Retrying in 15s...")
                time.sleep(15)
                continue
            return res
        except requests.RequestException as e:
            print(f"Network issue ({e}). Retrying in 15s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(15)
    return None


def ingest_hourly_bars(ticker):
    init_db()
    con = get_connection()
    sym_id = get_or_create_symbol(con, ticker)

    end_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=730)).strftime("%Y-%m-%d")

    latest_ts = get_latest_timestamp(con, sym_id)
    if latest_ts:
        start_date = latest_ts.split()[0]
        print(f"{ticker} has data up to {latest_ts}. Fetching from {start_date} to {end_date}...")

    if start_date > end_date:
        print(f"{ticker} is already up to date.")
        con.close()
        return

    ticker_dir = RAW_DATA_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = get_next_chunk_index(ticker_dir, "chunk")

    next_url = f"{BASE_URL}{ticker}/range/1/hour/{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000"

    try:
        while next_url:
            res = request_with_retry(next_url, params={"apiKey": API_KEY})
            if not res or res.status_code != 200:
                err_msg = res.text if res else "No response"
                print(f"Failed to fetch bars for {ticker}: {err_msg}")
                break

            data = res.json()
            results = data.get("results", [])
            if not results:
                break

            next_url = data.get("next_url")
            if next_url and next_url.startswith("https://api.polygon.io"):
                next_url = next_url.replace("https://api.polygon.io", "https://api.massive.com")

            chunk_file = ticker_dir / f"chunk_{chunk_idx:04d}.json"
            with open(chunk_file, "w") as f:
                json.dump(data, f, indent=2)

            bars = [
                {
                    "timestamp": datetime.datetime.fromtimestamp(b["t"] / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": b.get("o", 0.0),
                    "high": b.get("h", 0.0),
                    "low": b.get("l", 0.0),
                    "close": b.get("c", 0.0),
                    "volume": b.get("v", 0.0),
                    "transactions": b.get("n", 0)
                }
                for b in results if b.get("t")
            ]
            insert_hourly_bars(con, sym_id, bars)
            print(f"Saved chunk {chunk_idx} ({len(results)} bars) -> {chunk_file}")
            chunk_idx += 1

            if next_url:
                time.sleep(SLEEP_TIME)
    finally:
        con.close()


def ingest_ticker_fundamentals(ticker):
    init_db()
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT shares_outstanding, market_cap FROM symbols WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row and row[0] is not None and row[1] is not None:
        con.close()
        return

    url = f"https://api.massive.com/v3/reference/tickers/{ticker}"
    res = request_with_retry(url, params={"apiKey": API_KEY})
    if res and res.status_code == 200:
        data = res.json().get("results", {})
        shares = data.get("weighted_shares_outstanding") or data.get("share_class_shares_outstanding")
        mcap = data.get("market_cap")
        if shares and mcap:
            update_symbol_fundamentals(con, ticker, shares, mcap)
            print(f"Updated fundamentals for {ticker}: {shares:,} shares, ${mcap:,.0f} mcap")
    con.close()
