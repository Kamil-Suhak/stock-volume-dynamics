import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features import load_data, build_features
from src.db import get_connection


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    vol_true = np.exp(y_true)
    vol_pred = np.exp(y_pred)
    mape = np.mean(np.abs((vol_true - vol_pred) / vol_true)) * 100.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def store_predictions(con, pred_df):
    records = [
        (
            int(row['bar_id']),
            str(row['model_name']),
            float(row['actual_log_vol']),
            float(row['pred_log_vol']),
            float(row['residual'])
        )
        for _, row in pred_df.iterrows()
    ]
    con.executemany("""
        INSERT OR REPLACE INTO predictions (
            bar_id, model_name, actual_log_vol, pred_log_vol, residual
        ) VALUES (?, ?, ?, ?, ?);
    """, records)
    con.commit()


def store_benchmarks(con, benchmark_df):
    for _, row in benchmark_df.iterrows():
        con.execute("""
            INSERT INTO model_benchmarks (mode, model, cv_rmse, cv_mae, cv_mape, features_used)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (
            str(row["Mode"]),
            str(row["Model"]),
            float(row["CV_RMSE"]),
            float(row["CV_MAE"]),
            float(row["CV_MAPE_%"]),
            str(row.get("Features", ""))
        ))
    con.commit()


def evaluate_walk_forward(X, y_dev, df_clean, model, cv=5):
    tscv = TimeSeriesSplit(n_splits=cv)
    fold_metrics = []
    val_preds = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y_dev.iloc[train_idx]

        base_val = df_clean.iloc[val_idx]['base_intraday'].values
        y_val_actual = df_clean.iloc[val_idx]['target_log_vol'].values

        model.fit(X_train, y_train)
        pred_dev = model.predict(X_val)
        y_pred = base_val + pred_dev

        fold_metrics.append(calc_metrics(y_val_actual, y_pred))

        val_df = df_clean.iloc[val_idx].copy()
        val_df['actual_log_vol'] = y_val_actual
        val_df['pred_log_vol'] = y_pred
        val_df['residual'] = y_val_actual - y_pred
        val_preds.append(val_df)

    all_preds = pd.concat(val_preds, ignore_index=True).drop_duplicates(subset=['bar_id']).reset_index(drop=True)

    avg_metrics = {
        m: np.mean([f[m] for f in fold_metrics])
        for m in ["RMSE", "MAE", "MAPE"]
    }
    return avg_metrics, all_preds


def run_training(tickers=None, n_splits=5):
    print("Loading data from SQLite...")
    raw_df = load_data(tickers=tickers, rth_only=True)
    if raw_df.empty:
        print("No data found in database. Ingest data first.")
        return None

    print(f"Loaded {len(raw_df)} bars across {raw_df['ticker'].nunique()} tickers: {', '.join(raw_df['ticker'].unique())}")

    con = get_connection()
    modes = ["forward", "explanatory"]
    results_summary = []
    predictions_to_store = []

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=20.0))
        ]),
        "HistGradBoost": HistGradientBoostingRegressor(
            max_iter=120,
            loss="absolute_error",
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42
        )
    }

    for mode in modes:
        X, y, df_clean = build_features(raw_df, mode=mode)

        for model_label, model_obj in models.items():
            full_model_name = f"{mode.capitalize()}_{model_label}"

            metrics, test_preds = evaluate_walk_forward(
                X, y, df_clean, model_obj, cv=n_splits
            )

            results_summary.append({
                "Mode": mode.capitalize(),
                "Model": model_label,
                "CV_RMSE": metrics["RMSE"],
                "CV_MAE": metrics["MAE"],
                "CV_MAPE_%": metrics["MAPE"]
            })

            print(f"[{mode.capitalize()}] {model_label} -> RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")

            test_preds['model_name'] = full_model_name
            predictions_to_store.append(test_preds[['bar_id', 'model_name', 'actual_log_vol', 'pred_log_vol', 'residual']])

    summary_df = pd.DataFrame(results_summary)

    all_preds_df = pd.concat(predictions_to_store, ignore_index=True)
    store_predictions(con, all_preds_df)
    store_benchmarks(con, summary_df)
    con.close()

    print("\nBenchmark Results:")
    print(summary_df.to_string(index=False))
    print(f"\nStored {len(all_preds_df)} test predictions in database.")

    return summary_df


if __name__ == "__main__":
    run_training()
