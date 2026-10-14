import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db import init_db
from src.ingest import ingest_hourly_bars, ingest_ticker_fundamentals
from src.train import run_training

# Mag 7 + 3 AI Leaders
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AVGO",
    "AMD",
    "PLTR"
]


def main():
    print(f"Target tickers: {', '.join(TICKERS)}\n")
    init_db()

    print("Fetching market bars and fundamentals...")
    for idx, ticker in enumerate(TICKERS, 1):
        print(f"\n[{idx}/{len(TICKERS)}] Processing {ticker}...")
        try:
            ingest_hourly_bars(ticker)
            ingest_ticker_fundamentals(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("\nTraining models and evaluating...")
    run_training(tickers=TICKERS, n_splits=5)

    print("\nDone! Start dashboard with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
