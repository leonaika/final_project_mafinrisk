import pandas as pd
import numpy as np


class FeatGen():
    def __init__(self):
        pass

    def _hours_to_bars(self, h, bar_minutes):
        return int(h * 60 // bar_minutes)

    def _days_to_bars(self, d, bar_minutes):
        return int(d * (24 * 60 // bar_minutes))

    def _build_return_features(
        self,
        df: pd.DataFrame,
        stable_feats: list,
        price_col: str = "vwap",
        ticker_col: str = "ticker",
        timestamp_col: str = "timestamp",
        btc_ticker: str = "BTCUSDT",
        bar_minutes: int = 15,
    ) -> pd.DataFrame:
        """Compute only the return features present in stable_feats (and only their required lags)."""
        print("Building return features")

        # Ensure proper sort + price source
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False)
        df.sort_values([ticker_col, timestamp_col], inplace=True)

        if price_col not in df.columns:
            if {"amount", "volume"}.issubset(df.columns):
                df["vwap"] = (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
                df["vwap"] = df.groupby(ticker_col)["vwap"].ffill()
                price_col = "vwap"
            else:
                price_col = "close"

        # Which base returns do we need?
        need_hours = set()
        for name in stable_feats:
            if name.startswith("ret_") and "h" in name:
                # e.g. ret_24h or ret_24h_lag3
                h = int(name.split("_")[1].replace("h", ""))
                need_hours.add(h)

        # Compute each required base return
        for h in sorted(need_hours):
            w = self._hours_to_bars(h, bar_minutes)
            base = f"ret_{h}h"
            if base in stable_feats or any(f.startswith(base + "_lag") for f in stable_feats):
                df[base] = df.groupby(ticker_col)[price_col].pct_change(periods=w)

        # Compute only required lags
        for name in stable_feats:
            if name.startswith("ret_") and "_lag" in name:
                base, lagp = name.split("_lag")
                k = int(lagp)
                h = int(base.split("_")[1].replace("h", ""))
                w = self._hours_to_bars(h, bar_minutes)
                if base not in df.columns:
                    df[base] = df.groupby(ticker_col)[price_col].pct_change(periods=w)
                df[name] = df.groupby(ticker_col)[base].shift(k * w)

        return df

    def _build_regime_features(
        self,
        df: pd.DataFrame,
        stable_feats: list,
        ticker_col: str = "ticker",
        timestamp_col: str = "timestamp",
        price_col: str = "vwap",
        btc_ticker: str = "BTCUSDT",
        bar_minutes: int = 15,
    ) -> pd.DataFrame:
        """Compute only the regime features present in stable_feats, using minimal intermediates."""
        print("Building regime features")

        # Sort + price source
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False)
        df.sort_values([ticker_col, timestamp_col], inplace=True)

        if price_col not in df.columns:
            if {"amount", "volume"}.issubset(df.columns):
                df["vwap"] = (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
                df["vwap"] = df.groupby(ticker_col)["vwap"].ffill()
                price_col = "vwap"
            else:
                price_col = "close"

        # 1-bar return (temp) → needed for RV and correlations
        df["__ret1__"] = df.groupby(ticker_col)[price_col].pct_change()

        # ===== RV windows needed =====
        need_rv_days = set()
        if any(f in stable_feats for f in ["market_mean_rv_1d", "btc_rv_1d"]):
            need_rv_days.add(1)
        if any(f in stable_feats for f in ["market_mean_rv_7d", "btc_rv_7d", "rel_rv_7d_to_mkt"]):
            need_rv_days.add(7)
        # vol_short_change uses rv_1d and a 3d shift; dd_max_30d_voladj uses rv_3d
        if "vol_short_change" in stable_feats:
            need_rv_days.add(1)
        if "dd_max_30d_voladj" in stable_feats:
            need_rv_days.add(3)

        # compute needed rv_*d
        for d in sorted(need_rv_days):
            w = self._days_to_bars(d, bar_minutes)
            df[f"__rv_{d}d__"] = df.groupby(ticker_col)["__ret1__"].transform(
                lambda s: s.rolling(w, min_periods=w).std()
            )

        # market_mean_rv_*d
        if "market_mean_rv_1d" in stable_feats:
            m = df.groupby(timestamp_col)["__rv_1d__"].mean().rename("market_mean_rv_1d")
            df = df.merge(m, left_on=timestamp_col, right_index=True, how="left")
        if "market_mean_rv_7d" in stable_feats:
            m = df.groupby(timestamp_col)["__rv_7d__"].mean().rename("market_mean_rv_7d")
            df = df.merge(m, left_on=timestamp_col, right_index=True, how="left")

        # btc_rv_*d
        if "btc_rv_1d" in stable_feats:
            s = (
                df.loc[df[ticker_col] == btc_ticker, [timestamp_col, "__rv_1d__"]]
                .drop_duplicates(subset=[timestamp_col])
                .rename(columns={"__rv_1d__": "btc_rv_1d"})
            )
            df = df.merge(s, on=timestamp_col, how="left")
        if "btc_rv_7d" in stable_feats:
            s = (
                df.loc[df[ticker_col] == btc_ticker, [timestamp_col, "__rv_7d__"]]
                .drop_duplicates(subset=[timestamp_col])
                .rename(columns={"__rv_7d__": "btc_rv_7d"})
            )
            df = df.merge(s, on=timestamp_col, how="left")

        # rel_rv_7d_to_mkt
        if "rel_rv_7d_to_mkt" in stable_feats:
            if "market_mean_rv_7d" not in df.columns:
                m = df.groupby(timestamp_col)["__rv_7d__"].mean().rename("market_mean_rv_7d")
                df = df.merge(m, left_on=timestamp_col, right_index=True, how="left")
            df["rel_rv_7d_to_mkt"] = df["__rv_7d__"] / df["market_mean_rv_7d"]

        # vol_short_change = rv_1d / rv_1d.shift(3d)
        if "vol_short_change" in stable_feats:
            sh = self._days_to_bars(3, bar_minutes)
            df["vol_short_change"] = df["__rv_1d__"] / df.groupby(ticker_col)["__rv_1d__"].shift(sh)

        # ===== Breadth (20d, 50d, 100d) =====
        need_breadth = []
        for d in (20, 50, 100):
            if f"market_breadth_ma{d}d" in stable_feats:
                need_breadth.append(d)

        for d in need_breadth:
            w = self._days_to_bars(d, bar_minutes)
            ma = df.groupby(ticker_col)[price_col].transform(lambda s: s.rolling(w, min_periods=w).mean())
            flag = (df[price_col] > ma).astype(float)
            breadth = flag.groupby(df[timestamp_col]).mean().rename(f"market_breadth_ma{d}d")
            df = df.merge(breadth, left_on=timestamp_col, right_index=True, how="left")
            del ma, flag, breadth

        # ===== Trend slopes (ma_20d_slope_10d_pct, ma_50d_slope_20d_pct) =====
        if "ma_20d_slope_10d_pct" in stable_feats:
            w = self._days_to_bars(20, bar_minutes)
            sh = self._days_to_bars(10, bar_minutes)
            ma20 = df.groupby(ticker_col)[price_col].transform(lambda s: s.rolling(w, min_periods=w).mean())
            raw = ma20 - ma20.groupby(df[ticker_col]).shift(sh)
            df["ma_20d_slope_10d_pct"] = raw / ma20
            del ma20, raw

        if "ma_50d_slope_20d_pct" in stable_feats:
            w = self._days_to_bars(50, bar_minutes)
            sh = self._days_to_bars(20, bar_minutes)
            ma50 = df.groupby(ticker_col)[price_col].transform(lambda s: s.rolling(w, min_periods=w).mean())
            raw = ma50 - ma50.groupby(df[ticker_col]).shift(sh)
            df["ma_50d_slope_20d_pct"] = raw / ma50
            del ma50, raw

        # ===== Drawdowns (30d) — consistent & explicit =====
        need_dd30_cur = "dd_cur_30d" in stable_feats
        need_dd30_max = "dd_max_30d" in stable_feats or "dd_max_30d_voladj" in stable_feats
        need_dd30_vadj = "dd_max_30d_voladj" in stable_feats

        if need_dd30_cur or need_dd30_max or need_dd30_vadj:
            w30 = self._days_to_bars(30, bar_minutes)

            # rolling max of price (per ticker)
            roll_max = df.groupby(ticker_col)[price_col].transform(lambda s: s.rolling(w30, min_periods=w30).max())

            # current drawdown as a REAL column (not anonymous Series)
            df["__dd_cur_30d__"] = (df[price_col] / roll_max) - 1.0
            if need_dd30_cur:
                df["dd_cur_30d"] = df["__dd_cur_30d__"]

            # max drawdown over window = rolling min of current drawdown
            if need_dd30_max or need_dd30_vadj:
                df["__dd_max_30d__"] = df.groupby(ticker_col)["__dd_cur_30d__"].transform(
                    lambda s: s.rolling(w30, min_periods=w30).min()
                )
                if "dd_max_30d" in stable_feats:
                    df["dd_max_30d"] = df["__dd_max_30d__"]

            # vol-adjusted version — use the SAME definition as before
            # If your old code used rv_{vol_long_days}d (e.g., 3d), rebuild it explicitly here from the SAME return series.
            if need_dd30_vadj:
                vol_long_days = 3  # <-- set this to whatever you used previously
                wv = self._days_to_bars(vol_long_days, bar_minutes)

                # ensure 1-bar return source is identical to the rest of your pipeline
                if "__ret1__" not in df.columns:
                    df["__ret1__"] = df.groupby(ticker_col)[price_col].pct_change()

                df["__rv_long__"] = df.groupby(ticker_col)["__ret1__"].transform(
                    lambda s: s.rolling(wv, min_periods=wv).std()
                )

                df["dd_max_30d_voladj"] = df["__dd_max_30d__"] / df["__rv_long__"]

            # cleanup temps
            df.drop(
                columns=[c for c in ["__dd_cur_30d__", "__dd_max_30d__", "__rv_long__"] if c in df.columns],
                inplace=True,
                errors="ignore",
            )

        # ===== BTC correlation mean/std (1d, 3d, 7d, 14d) =====
        need_corr_days = []
        if any(f.endswith("_corr_btc_1d") for f in ["market_mean", "market_std"]):
            pass  # not used directly
        for d in (1, 3, 7, 14):
            if (f"market_mean_corr_btc_{d}d" in stable_feats) or (f"market_std_corr_btc_{d}d" in stable_feats):
                need_corr_days.append(d)

        # BTC 1-bar return (one row per timestamp)
        if need_corr_days:
            btc_ret = (
                df.loc[df[ticker_col] == btc_ticker, [timestamp_col, "__ret1__"]]
                .drop_duplicates(subset=[timestamp_col])
                .rename(columns={"__ret1__": "__btc_ret__"})
            )
            df = df.merge(btc_ret, on=timestamp_col, how="left")

        for d in sorted(set(need_corr_days)):
            w = self._days_to_bars(d, bar_minutes)

            # rolling corr per asset as a Series aligned to df index (temp only)
            corr_series = (
                df.groupby(ticker_col)
                .apply(lambda g: g["__ret1__"].rolling(w, min_periods=w).corr(g["__btc_ret__"]))
                .reset_index(level=0, drop=True)
            )
            # exclude BTC self-corr
            if (df[ticker_col] == btc_ticker).any():
                corr_series.loc[df.index[df[ticker_col] == btc_ticker]] = np.nan

            # aggregate by timestamp to mean and std, then merge
            mean_ts = corr_series.groupby(df[timestamp_col]).mean().rename(f"market_mean_corr_btc_{d}d")
            std_ts = corr_series.groupby(df[timestamp_col]).std(ddof=0).rename(f"market_std_corr_btc_{d}d")

            if f"market_mean_corr_btc_{d}d" in stable_feats:
                df = df.merge(mean_ts, left_on=timestamp_col, right_index=True, how="left")
            if f"market_std_corr_btc_{d}d" in stable_feats:
                df = df.merge(std_ts, left_on=timestamp_col, right_index=True, how="left")

            del corr_series, mean_ts, std_ts

        # ===== cleanup temps =====
        drop_tmp = [c for c in df.columns if c.startswith("__rv_") and c.endswith("__")]
        drop_tmp += ["__ret1__", "__btc_ret__"]
        df.drop(columns=[c for c in drop_tmp if c in df.columns], inplace=True, errors="ignore")

        return df
    
    def _trim_data(self, df: pd.DataFrame) -> pd.DataFrame:
        max_ts = df.groupby("ticker")["timestamp"].transform("max")
        cutoff = max_ts - pd.Timedelta(days=80)
        df = df.loc[df["timestamp"] >= cutoff].copy()
        return df

    def build_features(self, df):
        """
        After filtering only useful features during the analysis we calculate only them to meet RAM limits.
        Main groups of features are return features and market regime features
        """
        print('Building features')
        stable_feats = [
            "market_std_corr_btc_14d",
            "market_mean_corr_btc_14d",
            "market_mean_rv_7d",
            "btc_rv_7d",
            "market_std_corr_btc_7d",
            "market_mean_corr_btc_7d",
            "market_breadth_ma100d",
            "market_breadth_ma50d",
            "market_std_corr_btc_3d",
            "market_breadth_ma20d",
            "market_mean_rv_1d",
            "btc_rv_1d",
            "market_mean_corr_btc_3d",
            "market_mean_corr_btc_1d",
            "market_std_corr_btc_1d",
            "ret_6h",
            "ret_24h",
            "ret_12h",
            "ret_168h",
            "vol_short_change",
            "ma_20d_slope_10d_pct",
            "dd_max_30d",
            "ret_168h_lag2",
            "ret_168h_lag3",
            "ret_96h_lag2",
            "ret_96h",
            "ret_24h_lag3",
            "ret_96h_lag3",
            "dd_max_30d_voladj",
            "dd_cur_30d",
            "ma_50d_slope_20d_pct",
            "rel_rv_7d_to_mkt",
        ]
        # df = self._trim_data(df)
        df.reset_index(drop=True, inplace=True)
        df.sort_values(["ticker", "timestamp"], inplace=True)
        df["id"] = df["timestamp"].astype(str) + "_" + df["ticker"]

        df = self._build_return_features(df, stable_feats)
        df = self._build_regime_features(df, stable_feats)

        df.dropna(inplace=True)
        df.sort_values(["ticker", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df, stable_feats
