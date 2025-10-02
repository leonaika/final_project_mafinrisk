from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from sdk.oms_client import OmsClient, ApiError, DataSizeError
from constants import symbols_in_dataset
import pandas as pd
from tqdm.contrib.concurrent import thread_map
import os
from pathlib import Path

Candle = List[float]  # [ts, open, high, low, close, volume]
OMS_URL = "https://quant-competition-oms-test.yorkapp.com"
OMS_ACCESS_TOKEN = ""
client = OmsClient(OMS_URL, OMS_ACCESS_TOKEN)


def _fetch_one(
    client: OmsClient, symbol: str, timeframe: str, size: int
) -> Tuple[str, List[Candle]]:
    """Fetch OHLCV for a single symbol."""
    data = client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, size=size)
    return symbol, data


def fetch_many(
    symbols: List[str],
    timeframe: str = "15m",
    size: int = 500,
    max_workers: int = 3,  # keep small to play nice with rate limits
) -> Dict[str, List[Candle]]:
    """
    Fetch OHLCV for multiple symbols concurrently.
    Returns a dict: {symbol: [[ts, o, h, l, c, v], ...], ...}
    """
    results: Dict[str, List[Candle]] = {}
    errors: Dict[str, Exception] = {}

    with client:
        all_results = thread_map(
            _fetch_one,
            [client] * len(symbols),
            symbols,
            [timeframe] * len(symbols),
            [size] * len(symbols),
            max_workers=max_workers,
            desc="Fetching OHLCV data",
            unit="symbol",
        )

        # Process results
        for result in all_results:
            if isinstance(result, tuple) and len(result) == 2:
                sym, data = result
                results[sym] = data
            elif isinstance(result, Exception):
                print(f"[warn] Error: {result}")

    return results


# ---- Example “send to other modules” hook ----
def send_to_downstream(batch: Dict[str, List[Candle]]) -> None:
    """
    Replace this with your bus/queue/callback (e.g., model.predict(batch), pm.update(batch), etc.)
    For now, we just print a tiny summary.
    """
    for sym, rows in batch.items():
        if not rows:
            print(f"{sym}: no data")
            continue
        ts, o, h, l, c, v = rows[-1]
        print(f"{sym}: last={c} ts={ts} rows={len(rows)}")


def get_data(
    size=100,
    timeframe="15m",
    max_workers=32,
    csv_file="LIVETRADING/data/market_data_0121_0925.parquet",
):
    """
    Get market data, save to CSV, and load/update existing CSV if available

    Args:
        size: Number of candles to fetch
        timeframe: Timeframe for candles
        max_workers: Maximum parallel workers
        csv_file: Path to CSV file

    Returns:
        pd.DataFrame: Market data with OHLCV information
    """
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(csv_file):
        print(f"Loading existing data from {csv_file}")
        try:
            # Load existing data
            existing_df = pd.read_csv(
                csv_file,
                parse_dates=["timestamp"],
                dtype={
                    "ticker": str,
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float,
                },
            )

            latest_timestamps = existing_df.groupby("ticker")["timestamp"].max()

            print("Fetching new data...")
            symbols = client.get_symbols()
            symbols = pd.Series(symbols)

            assert (
                symbols.str.split("-").apply(len) == 3
            ).all(), "all tickers are expected in format TICK-USDT-PERP"
            symbols = symbols.str.split("-").apply(lambda x: x[0] + x[1])

            data = fetch_many(
                symbols, timeframe=timeframe, size=size, max_workers=max_workers
            )

            rows = []
            for ticker, candle_data in data.items():
                for candle in candle_data:
                    candle_timestamp = pd.to_datetime(candle[0], unit="ms")

                    # Only add candles that are newer than the latest timestamp for this ticker
                    if (
                        ticker not in latest_timestamps
                        or candle_timestamp > latest_timestamps[ticker]
                    ):
                        rows.append(
                            {
                                "ticker": ticker,
                                "timestamp": candle_timestamp,
                                "open": candle[1],
                                "high": candle[2],
                                "low": candle[3],
                                "close": candle[4],
                                "volume": candle[5],
                            }
                        )

            if rows:
                print(f"Adding {len(rows)} new candles to existing data")
                new_df = pd.DataFrame(rows)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(
                    subset=["ticker", "timestamp"]
                )
                combined_df = combined_df.sort_values(
                    ["ticker", "timestamp"]
                ).reset_index(drop=True)

                combined_df.to_csv(csv_file, index=False)
                print(f"Data updated and saved to {csv_file}")
                return combined_df
            else:
                print("No new data found. Returning existing data.")
                return existing_df

        except Exception as e:
            print(f"Error loading existing CSV: {e}. Fetching fresh data...")

    # If no existing file or error loading, fetch fresh data
    df = get_data_simple(size, timeframe, max_workers, csv_file)

    return df


def get_data_simple(
    size=100, timeframe="15m", max_workers=32, csv_file="data/market_data.csv"
):
    """
    Simplified version that always fetches fresh data and overwrites CSV
    """
    print("Getting fresh data")

    symbols = client.get_symbols()
    symbols = pd.Series(symbols)

    assert (
        symbols.str.split("-").apply(len) == 3
    ).all(), "all tickers are expected in format TICK-USDT-PERP"
    symbols = symbols.str.split("-").apply(lambda x: x[0] + x[1])

    data = fetch_many(symbols, timeframe=timeframe, size=size, max_workers=max_workers)

    rows = []
    for ticker, candle_data in data.items():
        for candle in candle_data:
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": pd.to_datetime(candle[0], unit="ms"),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                }
            )

    df = pd.DataFrame(rows)

    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")

    return df


if __name__ == "__main__":
    # Example usage
    data = fetch_many(symbols_in_dataset, timeframe="15m", size=1, max_workers=3)
    send_to_downstream(data)
