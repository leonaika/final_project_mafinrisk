from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import md

from sdk.oms_client import OmsClient
from constants import coins_to_trade

# ---------- Config (tweak here) ----------
TOP_K_LONG = 10
TOP_K_SHORT = 10

MAX_GROSS_LEVERAGE = 1.5                # 150% of NAV
PER_ASSET_CAP_USDT = 3000               # per-asset cap
HYSTERESIS_PCT = 0.20                   # only trade if |delta| > 20% of target
MIN_NOTIONAL_USDT = 50                  # ignore dust adjustments

# Vol sizing config (15m bars)
VOL_LOOKBACK_BARS = 192                 # ~2 days (2 * 24 * 4)
VOL_EWMA_ALPHA = 0.10                   # emphasize recent vol
VOL_FLOOR = 1e-4                        # avoid div-by-zero

# ---------- Config (tweak here) ----------
STOP_K = 4  # close if adverse move > k * sigma from entry (vol stop)
# ----------------------------------------


@dataclass
class TargetPos:
    instrument_name: str
    instrument_type: str   # "future"
    target_value: float    # positive notional in USDT
    position_side: str     # "LONG" or "SHORT"


class PortfolioManager:
    def __init__(self, client: OmsClient, simulation_flag: bool = False):
        self.client = client

        if simulation_flag:
            # self.balance = 10000
            self.asset_qty = {'USDT': 10000}
        self._entries = {}
            

    # ---------- Helpers ----------
    def _format_instrument(self, symbol: str) -> str:
        """
        Convert 'BTCUSDT' -> 'BTC-USDT-PERP'
        """
        usdt_idx = symbol.find("USDT")
        base = symbol[:usdt_idx]
        quote = symbol[usdt_idx:]
        return f"{base}-{quote}-PERP"

    def _current_position_usdt(self) -> Dict[str, float]:
        """
        Map instrument_name -> absolute notional in USDT for the current open position.

        Expects self.client.get_position() to return dicts with:
          'instrument_name', 'position_side', 'quantity', 'avg_price', 'value'
        We prefer 'value' (signed). We store ABS(value) to compare against positive target_value.
        """
        pos_map: Dict[str, float] = {}
        try:
            positions = self.client.get_position()
            if not positions:
                return pos_map

            for p in positions:
                name = p.get("instrument_name")
                if not name:
                    continue

                # Prefer explicit 'value' if present
                val = p.get("value")
                if val is not None:
                    try:
                        abs_notional = abs(float(val))
                    except (TypeError, ValueError):
                        abs_notional = 0.0
                else:
                    # Fallback: |qty| * avg_price
                    qty = p.get("quantity", 0.0)
                    px = p.get("avg_price", 0.0)
                    try:
                        abs_notional = abs(float(qty)) * float(px)
                    except (TypeError, ValueError):
                        abs_notional = 0.0

                if abs_notional <= 0.0:
                    continue

                # If multiple entries for same instrument, keep max notionals (defensive for hysteresis)
                prev = pos_map.get(name, 0.0)
                pos_map[name] = max(prev, abs_notional)

        except Exception as e:
            print(f"_current_position_usdt error: {e}")

        return pos_map

    def _estimate_vols(self, data: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
        """
        Compute per-ticker EWMA std of log returns over the recent VOL_LOOKBACK_BARS.
        Returns dict: ticker -> per-bar vol (not annualized).
        `data` columns: ['timestamp', 'ticker', 'close', ...]
        """
        if data is None or data.empty:
            return {}

        df = data.loc[data["ticker"].isin(tickers), ["timestamp", "ticker", "close"]].dropna()
        if df.empty:
            return {}

        # Ensure sorted for each ticker
        df = df.sort_values(["ticker", "timestamp"])

        # 15m log returns
        df["lr"] = np.log(df["close"]).groupby(df["ticker"]).diff()

        # Keep only recent window per ticker
        df = df.groupby("ticker", group_keys=False).tail(VOL_LOOKBACK_BARS)

        vols: Dict[str, float] = {}
        for sym, grp in df.groupby("ticker"):
            lr = grp["lr"].dropna()
            if lr.empty:
                continue
            ewma_var = lr.ewm(alpha=VOL_EWMA_ALPHA, adjust=False).var(bias=False)
            v = float(np.sqrt(max(float(ewma_var.iloc[-1]), 0.0)))
            vols[sym] = max(v, VOL_FLOOR)
        return vols

    def _allocate_inverse_vol(
        self,
        side_budget: float,
        tickers: List[str],
        vols: Dict[str, float],
        per_asset_cap: float
    ) -> Dict[str, float]:
        """
        Allocate dollars on one side (long OR short) proportional to 1/vol,
        with a per-asset dollar cap. Returns dict: ticker -> dollars (>=0).
        Falls back to equal-dollar if any ticker on the side lacks vol.
        """
        if side_budget <= 0 or not tickers:
            return {}

        missing = [t for t in tickers if t not in vols]
        if len(missing) > 0:
            # Robust fallback: equal-dollar allocation
            per = min(per_asset_cap, side_budget / len(tickers))
            return {t: per for t in tickers}

        inv_vol = {t: 1.0 / max(vols[t], VOL_FLOOR) for t in tickers}
        s = sum(inv_vol.values())
        if s <= 0:
            per = min(per_asset_cap, side_budget / len(tickers))
            return {t: per for t in tickers}

        raw = {t: side_budget * inv_vol[t] / s for t in tickers}
        # Apply per-asset cap
        capped = {t: min(per_asset_cap, raw[t]) for t in tickers}
        total_after_cap = sum(capped.values())

        # If over side budget (rounding/caps), scale down proportionally
        if total_after_cap > side_budget and total_after_cap > 0:
            scale = side_budget / total_after_cap
            capped = {t: v * scale for t, v in capped.items()}

        # Drop dust
        capped = {t: v for t, v in capped.items() if v >= MIN_NOTIONAL_USDT}
        return capped

    # ---------- Strategy steps ----------
    def select_positions(self, predictions: pd.DataFrame) -> Dict[str, List[str]]:
        """
        predictions: DataFrame with columns ['ticker', 'predict', ...]
        """
        score_col = "predict"
        df = predictions[predictions["ticker"].isin(coins_to_trade)].copy()
        if df.empty:
            return {"long": [], "short": []}

        df = df.sort_values(score_col, ascending=False)
        longs = list(df.head(TOP_K_LONG)["ticker"])
        shorts = list(df.tail(TOP_K_SHORT)["ticker"])
        return {"long": longs, "short": shorts}
    
    def _trim_data(self, df: pd.DataFrame) -> pd.DataFrame:
        max_ts = df.groupby("ticker")["timestamp"].transform("max")
        cutoff = max_ts - pd.Timedelta(days=80)
        df = df.loc[df["timestamp"] >= cutoff].copy()
        return df

    def get_weights(self, positions: Dict[str, List[str]], simulation: bool = False, data: pd.DataFrame = None) -> List[Dict]:
        """
        Build target positions that:
        - split gross 50/50 long vs short (neutral),
        - obey per-asset cap ($3k) and gross leverage cap (<= 150% * NAV),
        - allocate dollars inverse to volatility (EWMA of 15m returns),
        - re-equalize long/short gross after caps,
        - fall back to equal-dollar if vol data missing.
        """
        # NAV from available balance (you can enhance to include unrealized PnL if API allows)
        if simulation:
            nav = self.get_nav_simulation(data)
        else:
            nav = float(self.client.get_balance()[0]["balance"])
        nL, nS = len(positions["long"]), len(positions["short"])
        if nL == 0 and nS == 0:
            return []

        gross_budget = MAX_GROSS_LEVERAGE * nav

        # Base 50/50 split (if one side is empty, give it all to the other)
        long_budget = gross_budget * 0.5 if nL > 0 else 0.0
        short_budget = gross_budget * 0.5 if nS > 0 else 0.0
        if nL == 0 and nS > 0:
            short_budget = gross_budget
        if nS == 0 and nL > 0:
            long_budget = gross_budget

        # --- Volatility-scaled allocation ---
        if simulation:
            pass
        else:
            data = md.get_data()[['timestamp', 'ticker', 'close']].copy()
            data = self._trim_data(data)
        tickers = positions["long"] + positions["short"]
        vols = self._estimate_vols(data, tickers)

        long_dollars = self._allocate_inverse_vol(long_budget, positions["long"], vols, PER_ASSET_CAP_USDT)
        short_dollars = self._allocate_inverse_vol(short_budget, positions["short"], vols, PER_ASSET_CAP_USDT)

        # Actual gross on each side after caps
        long_gross = sum(long_dollars.values())
        short_gross = sum(short_dollars.values())

        # Re-equalize both sides to the smaller gross (keeps neutrality even after caps)
        if long_gross > 0 and short_gross > 0:
            target_side_gross = min(long_gross, short_gross)
            if long_gross > target_side_gross:
                scale = target_side_gross / long_gross
                long_dollars = {t: v * scale for t, v in long_dollars.items()}
                long_gross = target_side_gross
            if short_gross > target_side_gross:
                scale = target_side_gross / short_gross
                short_dollars = {t: v * scale for t, v in short_dollars.items()}
                short_gross = target_side_gross

        # Safety: ensure combined gross ≤ gross_budget
        total_gross = long_gross + short_gross
        if total_gross > gross_budget and total_gross > 0:
            scale = gross_budget / total_gross
            long_dollars = {t: v * scale for t, v in long_dollars.items()}
            short_dollars = {t: v * scale for t, v in short_dollars.items()}

        # Convert to API dicts
        target_positions: List[TargetPos] = []
        for coin, usd in long_dollars.items():
            if usd < MIN_NOTIONAL_USDT:
                continue
            instrument_name = self._format_instrument(coin)
            target_positions.append(TargetPos(
                instrument_name=instrument_name,
                instrument_type="future",
                target_value=float(round(usd, 2)),
                position_side="LONG"
            ))
        for coin, usd in short_dollars.items():
            if usd < MIN_NOTIONAL_USDT:
                continue
            instrument_name = self._format_instrument(coin)
            target_positions.append(TargetPos(
                instrument_name=instrument_name,
                instrument_type="future",
                target_value=float(round(usd, 2)),
                position_side="SHORT"
            ))

        # Final clamp in case rounding pushed us over
        total_gross = sum(tp.target_value for tp in target_positions)
        if total_gross > gross_budget and total_gross > 0:
            scale = gross_budget / total_gross
            for tp in target_positions:
                tp.target_value = float(round(tp.target_value * scale, 2))

        # Return as list of dicts for API
        out = [{
            "instrument_name": tp.instrument_name,
            "instrument_type": tp.instrument_type,
            "target_value": tp.target_value,
            "position_side": tp.position_side
        } for tp in target_positions]

        return out

    def _apply_hysteresis(self, targets: List[Dict]) -> List[Dict]:
        """
        Only send targets that differ from current by > HYSTERESIS_PCT and ≥ MIN_NOTIONAL_USDT in absolute delta.
        Compares ABS current notional vs target_value (both positive).
        """
        cur = self._current_position_usdt()  # name -> abs notional
        filtered: List[Dict] = []
        for t in targets:
            name = t["instrument_name"]
            tgt = float(t["target_value"])
            cur_abs = float(cur.get(name, 0.0))
            if tgt <= 0:
                continue
            delta = abs(tgt - cur_abs)
            if (delta / max(tgt, 1e-6)) >= HYSTERESIS_PCT and delta >= MIN_NOTIONAL_USDT:
                filtered.append(t)
        return filtered
    
    def _enforce_live_stops(self, targets: List[Dict]) -> List[Dict]:
        """
        Check current open positions via client.get_position() and
        force target_value = 0.0 for any instrument that breached STOP_K * sigma band.
        Uses the latest md.get_data() snapshot for price/sigma.
        """
        if not targets:
            return targets

        # latest prices per ticker
        mdf = md.get_data()[["timestamp","ticker","close"]].copy()
        if mdf.empty:
            return targets
        mdf = mdf.sort_values(["ticker","timestamp"])
        last_px = mdf.groupby("ticker")["close"].last().to_dict()

        # current open positions
        open_pos = self.client.get_position() or []
        open_tickers = []
        side_by_tk = {}
        for p in open_pos:
            name = p.get("instrument_name","")
            if "-USDT" not in name:
                continue
            base = name.split("-USDT")[0]
            tk = f"{base}USDT"
            qty = float(p.get("quantity", 0) or 0)
            if abs(qty) <= 0:
                continue
            open_tickers.append(tk)
            side_by_tk[tk] = "LONG" if qty > 0 else "SHORT"

        if not open_tickers:
            return targets

        # per-ticker sigma from recent window
        vols = self._estimate_vols(mdf, open_tickers)

        # use entry state if you have it; otherwise approximate entry at current price (no stop)
        # here we do a price-band check relative to average price (fallback)
        forced_flat = set()
        for p in open_pos:
            name = p.get("instrument_name","")
            if "-USDT" not in name:
                continue
            base = name.split("-USDT")[0]
            tk = f"{base}USDT"
            if tk not in side_by_tk:
                continue

            px_now = float(last_px.get(tk, np.nan))
            if not np.isfinite(px_now):
                continue

            # Prefer your stored entry if available (from simulator/live state), else use avg_price
            ent = self._entries.get(tk)
            if ent:
                entry_px = float(ent["entry_px"])
                side = ent["side"]
                sigma = float(vols.get(tk, VOL_FLOOR))
            else:
                entry_px = float(p.get("avg_price", px_now) or px_now)
                side = side_by_tk[tk]
                sigma = float(vols.get(tk, VOL_FLOOR))

            band = STOP_K * sigma * entry_px
            if (side == "LONG"  and px_now <= entry_px - band) or \
            (side == "SHORT" and px_now >= entry_px + band):
                forced_flat.add(tk)

        if not forced_flat:
            return targets

        # Override target_value for violators to flat (0)
        out = []
        for t in targets:
            inst = t["instrument_name"]
            base = inst.split("-USDT")[0] if "-USDT" in inst else ""
            tk = f"{base}USDT" if base else ""
            if tk in forced_flat:
                t = {**t, "target_value": 0.0}
            out.append(t)

        # clear entry state for flattened names
        for tk in forced_flat:
            self._entries.pop(tk, None)

        return out
    
    def _validate_predictions(self, predictions: pd.DataFrame):
        if predictions is None or predictions.empty:
            raise ValueError("manage(): predictions is empty.")
        if "ticker" not in predictions.columns or "predict" not in predictions.columns:
            raise ValueError("manage(): predictions must have ['ticker','predict'] columns.")

    def manage(self, predictions: pd.DataFrame):
        """
        Live entry: validate predictions, build targets, (optionally) enforce live stops,
        apply hysteresis, and submit batch.
        """
        # 0) Validate inputs
        self._validate_predictions(predictions)

        # 1) Select and size
        positions = self.select_positions(predictions)
        targets = self.get_weights(positions)  # uses live balance + md.get_data()

        if not targets:
            print("No targets (empty selection/sizing).")
            return {"status": "skipped", "reason": "no_targets"}

        # 2) (Optional) enforce live stop-loss before hysteresis
        try:
            targets = self._enforce_live_stops(targets)
        except Exception as e:
            print(f"live stop check skipped due to error: {e}")

        # 3) Hysteresis vs current book
        targets = self._apply_hysteresis(targets)
        if not targets:
            print("No meaningful rebalance needed")
            return {"status": "skipped", "reason": "hysteresis"}

        # 4) Submit to OMS (with a small safety wrapper)
        try:
            print("Submitting targets:", targets)
            # res = self.client.set_target_position_batch(targets)
            return 0
        except Exception as e:
            print(f"set_target_position_batch error: {e}")
            return {"status": "error", "error": str(e)}

        print("Response:", res)
        return res

    
    def get_nav_simulation(self, data):
        prices = data[data['ticker'].isin(self.asset_qty.keys())].drop_duplicates('ticker', keep='last')
        nav = 0
        for asset in self.asset_qty:
            if asset == 'USDT':
                price = 1
            else:
                price = prices.loc[prices['ticker'] == asset]['close'].iloc[0]
            value = price * self.asset_qty[asset]
            nav += value
        return nav
    
    def manage_simulator(self, predictions: pd.DataFrame = None, data: pd.DataFrame = None, md_trigger_flag: bool = False):
        def inst_to_ticker(inst_name: str) -> str:
            if not inst_name or "-USDT" not in inst_name:
                return ""
            base = inst_name.split("-USDT")[0]
            return f"{base}USDT"

        # 1) Build positions from signal as usual
        positions = self.select_positions(predictions)
        targets = self.get_weights(positions, True, data)

        if not targets:
            print("No meaningful rebalance needed")
            return {"status": "skipped", "reason": "hysteresis"}

        # 2) Compute current NAV and a quick price/sigma snapshot
        cur_nav = self.get_nav_simulation(data)

        # prices: last close per ticker
        last_prices = (
            data[["ticker", "close"]]
            .dropna()
            .groupby("ticker", as_index=True)
            .last()["close"]
            .to_dict()
        )

        # current open tickers (exclude USDT and zero qty)
        open_tickers = [t for t, q in self.asset_qty.items() if t != "USDT" and abs(q) > 0]

        # σ snapshot for open tickers (use your existing estimator)
        vols = self._estimate_vols(
            data[["timestamp", "ticker", "close"]].copy(),
            open_tickers
        )

        # 3) Volatility stop: if adverse move > STOP_K * sigma, force-flat
        forced_flat = set()
        for tk in open_tickers:
            px_now = float(last_prices.get(tk, np.nan))
            if not np.isfinite(px_now):
                continue

            # ensure we have an entry state (if sim started with a position)
            ent = self._entries.get(tk)
            if ent is None:
                # initialize lazily from current snapshot
                side = "LONG" if self.asset_qty.get(tk, 0.0) > 0 else "SHORT"
                sig = float(vols.get(tk, VOL_FLOOR))
                self._entries[tk] = {"entry_px": px_now, "sigma": sig, "side": side}
                ent = self._entries[tk]

            entry_px = float(ent["entry_px"])
            entry_sig = float(ent["sigma"])
            side = ent["side"]

            # price-based band using per-bar sigma
            band = STOP_K * entry_sig * entry_px  # σ in price terms
            stop_long = (side == "LONG") and (px_now <= entry_px - band)
            stop_short = (side == "SHORT") and (px_now >= entry_px + band)
            if stop_long or stop_short:
                forced_flat.add(tk)

        # 4) Apply stops by overriding targets for those tickers to zero
        if forced_flat:
            for t in targets:
                tk = inst_to_ticker(t["instrument_name"])
                if tk in forced_flat:
                    t["target_value"] = 0.0  # flatten
                    print('FLAT')
            # also clear their entry state—they will reinitialize on fresh entry
            for tk in forced_flat:
                self._entries.pop(tk, None)

        # 5) Rebuild holdings from targets (same as your logic)
        coins_value = 0.0
        new_qty = {}

        for t in targets:
            inst = t["instrument_name"]
            tk = inst_to_ticker(inst)
            if not tk:
                continue
            px = float(last_prices.get(tk, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue

            tv = float(t["target_value"])
            if tv <= 0:
                new_qty[tk] = 0.0
                continue

            side = (t["position_side"] or "").upper()
            q = (tv / px) * (1.0 if side == "LONG" else -1.0)
            new_qty[tk] = round(q, 5)
            coins_value += tv if side == "LONG" else -tv

        # USDT residual (keeps total NAV consistent)
        new_qty["USDT"] = cur_nav - coins_value

        # 6) Update entry state ONLY for lines newly opened or flipped side
        for tk, q in new_qty.items():
            if tk == "USDT":
                continue
            old_q = self.asset_qty.get(tk, 0.0)
            old_side = "LONG" if old_q > 0 else "SHORT" if old_q < 0 else "FLAT"
            new_side = "LONG" if q > 0 else "SHORT" if q < 0 else "FLAT"

            if new_side == "FLAT":
                # closed -> drop entry state
                self._entries.pop(tk, None)
                continue

            px_now = float(last_prices.get(tk, np.nan))
            if not np.isfinite(px_now):
                continue

            # If we were flat and now non-flat, or if the side flipped, (re)initialize entry
            if old_side == "FLAT" and new_side != "FLAT":
                sig_now = float(self._estimate_vols(data[["timestamp", "ticker", "close"]], [tk]).get(tk, VOL_FLOOR))
                self._entries[tk] = {"entry_px": px_now, "sigma": sig_now, "side": new_side}
            elif old_side != new_side:
                sig_now = float(self._estimate_vols(data[["timestamp", "ticker", "close"]], [tk]).get(tk, VOL_FLOOR))
                self._entries[tk] = {"entry_px": px_now, "sigma": sig_now, "side": new_side}
            # else: keep existing entry (same side, still open)

        # 7) Commit new holdings
        self.asset_qty = new_qty

        return {
            "status": "ok",
            "balance": float(self.get_nav_simulation(data)),
            "positions": {k: float(v) for k, v in self.asset_qty.items()}
        }

