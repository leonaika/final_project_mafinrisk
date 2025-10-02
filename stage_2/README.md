# Bull in the Bushes strategy: Regime-Invariant Feature Selection

## 1. Overview
Our solution is designed to build **robust return-prediction models** that perform consistently across **different market regimes** by selecting stable features, rather than overfitting to a specific period or volatility state.
The key innovation is a **regime-invariant feature selection** pipeline that identifies and retains only those features that remain predictive in various market conditions.

---

## 2. Feature Engineering

We generate two groups of features:

### 2.1 Return-based features
- Rolling simple returns over multiple horizons (6h, 12h, 24h, 96h, 168h)
- Lags of each return series (up to 3 lags), ensuring no leakage

### 2.2 Regime-describing features
These features capture the **state of the market** at a point in time:
- **Volatility**: realized vol (1d, 3d, 7d, 14d), volatility ratios (e.g., 1d/3d, 3d/14d), short-vol change
- **Trend**: slopes of moving averages over different horizons, % of positive bars (trend stability)
- **Market breadth**: % of tickers above MA20, MA50, MA100
- **Correlation to BTC**: rolling correlations metrics over different windows
- **Relative volatility**: vs market average and vs BTC
- **Drawdowns**: current and max over 14d and 30d, including volatility-adjusted versions
- **Regime persistence**: portion of time spent in high/low volatility states over the past 7 days

### 2.3 Volume features
We did not use the volume based features for two reasons:
- According to the code submission requirements we could not modify data preparation function (`get_all_symbol_kline`). This function does not generate volume feature however it is available in train data.
- We also did some experiments with volume features and came to conclusion these features are negligible for an overall score.

---

## 3. Regime-Invariant Feature Selection - Core Idea

We've started with a baseline model which was trained using sliding-window cross validation. Window size was 150 days with last 30 days for validation, the window step was 30 days. It performed well on some windows and poorly on other windows. We observed that the model was highly sensitive to the specific time window it was trained on. Since the provided data spanned across four years, we concluded that the cryptocurrency market likely shifted multiple times between distinct states, or "regimes," such as periods of high and low volatility. This indicated that we should focus on training a more robust model capable of performing well across these various market conditions.

The main hypothesis is that **features predictive in one market regime may fail in another** and this phenomenon affects the model perfomance. 

To address this we define market regimes, split dataset into distinct regime modes and keep only those features that remain useful in most regimes.

### Step-by-step process

#### 3.1 Defining regimes
We select **high-level market state indicators** (regime determinators) that we assume strongly affect signal behavior. We also make sure that these regime indicators describe different 'types' of regimes:
- `market_breadth_ma100d` - overall trend participation, percent of coins that are above MA 100d
- `market_mean_corr_btc_14d` - connection to main crypto driver (BTC)
- `vol_ratio_3d_14d` - volatility type regime
- `ma_50d_slope_20d_pct` - medium-term trend strength

Each indicator is **binned into categories** (low/high) via median cuts. This way we get $2^4 = 16$ regimes of the market (2 types, 4 features)

#### 3.2 Per-regime model fitting
For each sufficiently large regime bucket:
1. Train a **small CatBoostRegressor** only on data within that regime.
2. Compute feature importances using **PredictionValuesChange** (change in predictions when feature is permuted)

#### 3.3 Stability scoring
- For each feature, record whether it appears in the **top-N** importances for a given regime.
- Compute **`freq_in_topN`** = proportion of regimes where it appears in the top-N.
- Keep only features with `freq_in_topN ≥ threshold`

In other words we only kept those features, which were in top 40 features at least in 60% of regimes ($16*0.6=12$). In our case we decreased number of features **from ~90 to 32**.

#### 3.4 Result
We obtain a **`stable_feats`** list – features that are **predictive across a variety of market conditions**.  
These are the only features passed to the final model, ensuring **regime robustness** and reducing the risk of overfitting to one particular time period.

---

## 4. Final Model Training
- **Model**: CatBoostRegressor
- **Categoricals**: categorical regime features are passed directly to CatBoost for automatic handling
- **Training**: full historical dataset with only `stable_feats`
- **Validation**: after selecting `stable_feats` we ran another sliding window cross-validation to determine model parameters

---

## 5. Why This Works
- **Explicit market awareness**: Rather than hoping regularization alone will handle market shifts, we measure and enforce feature stability across market states.
- **Reduced overfitting**: The model only sees features that have demonstrated predictive power in multiple, distinct market environments.
- **Interpretability**: The regime determinators are an interpretable set of signals which are economically intuitive and directly describe the underlying structure of the market, allowing us to understand why it performs differently in various conditions.

---

## 6. Novelty
Most approaches treat the entire training history as one homogenous dataset.  
We explicitly:
1. **Split history into market regimes** using high-level descriptors
2. **Independently measure feature value** in each regime
3. **Keep only features that consistently matter**, ensuring robustness to unseen future regimes

This method bridges **quantitative feature engineering** and **robust ML selection**, aiming for stable performance rather than short-term optimization.

## 7. Anticipated results
We anticipate in-sample score of 0.77. We also used last 3 months of available data as a test dataset. We have measured that the score value is decreasing as we increase the horizon of the test data. For 1 month it was ~0.2, for 2 months ~0.1 etc. Which makes sense, because the predictive power of model decreases as the market evolves into new shapes of regimes that were not present or detected in the training period. Which is why in reality we would recommend to continously tune the model and use it for short horizon – up to 1 month.
