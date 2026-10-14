import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            shares_outstanding INTEGER,
            market_cap REAL
        );
    """)
    try:
        cur.execute("ALTER TABLE symbols ADD COLUMN shares_outstanding INTEGER;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE symbols ADD COLUMN market_cap REAL;")
    except Exception:
        pass
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hourly_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            transactions INTEGER NOT NULL,
            UNIQUE(symbol_id, timestamp),
            FOREIGN KEY (symbol_id) REFERENCES symbols(id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bar_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            actual_log_vol REAL NOT NULL,
            pred_log_vol REAL NOT NULL,
            residual REAL NOT NULL,
            UNIQUE(bar_id, model_name),
            FOREIGN KEY (bar_id) REFERENCES hourly_bars(id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            model TEXT NOT NULL,
            cv_rmse REAL NOT NULL,
            cv_mae REAL NOT NULL,
            cv_mape REAL NOT NULL,
            features_used TEXT,
            trained_at TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()
    con.close()

def get_or_create_symbol(con, ticker):
    cur = con.cursor()
    cur.execute("SELECT id FROM symbols WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO symbols (ticker) VALUES (?)", (ticker,))
    con.commit()
    return cur.lastrowid

def insert_hourly_bars(con, symbol_id, bars):
    cur = con.cursor()
    cur.executemany("""
        INSERT OR IGNORE INTO hourly_bars (symbol_id, timestamp, open, high, low, close, volume, transactions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [(symbol_id, b["timestamp"], b["open"], b["high"], b["low"], b["close"], b["volume"], b["transactions"]) for b in bars])
    con.commit()

def get_latest_timestamp(con, symbol_id):
    cur = con.cursor()
    cur.execute("SELECT MAX(timestamp) FROM hourly_bars WHERE symbol_id = ?", (symbol_id,))
    row = cur.fetchone()
    return row[0] if row else None

def update_symbol_fundamentals(con, ticker, shares_outstanding, market_cap):
    cur = con.cursor()
    cur.execute("""
        UPDATE symbols SET shares_outstanding = ?, market_cap = ? WHERE ticker = ?;
    """, (shares_outstanding, market_cap, ticker))
    con.commit()
