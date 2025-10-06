"""
Daily auto-selecting portfolio from 50 stocks using:
- ML model to predict next-day log returns (mu)
- Ledoit–Wolf covariance (Sigma)
- Convex mean–variance optimizer with turnover & cost penalties

Adapted for the FYP-FinAgent dataset structure.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
import cvxpy as cp
import os
import glob
from typing import Tuple, List

# ---------------------------------------------------------------------
# 1) Data loading for FYP-FinAgent dataset
# ---------------------------------------------------------------------

def load_data(data_dir: str = "processed_data") -> pd.DataFrame:
    """
    Load all CSV files from processed_data directory and combine into panel format.
    Returns DataFrame with columns: ['date', 'ticker', 'close', 'volume', ...features...]
    """
    print("Loading data from processed_data directory...")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for file_path in csv_files:
        # Extract ticker from filename (e.g., "ADANIPORTS_aligned.csv" -> "ADANIPORTS")
        ticker = os.path.basename(file_path).replace("_aligned.csv", "")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Convert first column (date) to datetime
            df['date'] = pd.to_datetime(df.iloc[:, 0])
            
            # Drop the unnamed index column
            df = df.drop(df.columns[0], axis=1)
            
            # Clean data - convert all columns to numeric where possible
            for col in df.columns:
                if col not in ['date', 'ticker']:
                    # Try to convert to numeric, replace non-numeric with NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            all_data.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded successfully")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    combined_df = combined_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['date', 'ticker']).reset_index(drop=True)
    
    print(f"Combined dataset: {len(combined_df)} rows, {combined_df['ticker'].nunique()} tickers")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Available columns: {len(combined_df.columns)}")
    
    return combined_df

# ---------------------------------------------------------------------
# 2) Feature configuration based on your dataset
# ---------------------------------------------------------------------

def get_feature_candidates(df: pd.DataFrame) -> List[str]:
    """Get available features from non_static_columns.txt, excluding metadata columns."""
    
    # Read non_static_columns.txt
    try:
        with open('non_static_columns.txt', 'r') as f:
            non_static_features = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("non_static_columns.txt not found, using all available features")
        non_static_features = []
    
    # Metadata columns to exclude
    exclude_cols = {'date', 'ticker', 'close', 'volume', 'open', 'high', 'low', 'vwap', 'value_traded', 'total_trades'}
    
    # Get features that exist in both the dataset and non_static_columns.txt
    available_cols = [col for col in df.columns if col not in exclude_cols]
    
    if non_static_features:
        # Use intersection of non_static_features and available columns
        feature_candidates = [col for col in non_static_features if col in available_cols]
    else:
        # Fallback to all available columns
        feature_candidates = available_cols
    
    # Filter out columns that are mostly NaN or constant
    filtered_features = []
    for col in feature_candidates:
        if df[col].notna().sum() > len(df) * 0.1:  # At least 10% non-null values
            if df[col].nunique() > 1:  # Not constant
                filtered_features.append(col)
    
    print(f"Using {len(filtered_features)} features from non_static_columns.txt (out of {len(non_static_features)} specified)")
    return filtered_features

# ---------------------------------------------------------------------
# 3) Target construction
# ---------------------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-day log return per ticker as target 'y_next'."""
    df = df.copy()
    
    # Clean the close column - convert to numeric and handle any non-scalar values
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Remove rows where close is NaN
    df = df.dropna(subset=['close'])
    
    # Calculate log returns per ticker
    df['log_close'] = np.log(df['close'])
    
    # Calculate next-day log returns using groupby
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['y_next'] = df.groupby('ticker')['log_close'].diff(-1)  # diff(-1) gives next day - current day
    
    return df

# ---------------------------------------------------------------------
# 4) Feature selection
# ---------------------------------------------------------------------

def select_features_time_aware(train_df: pd.DataFrame,
                               candidate_cols: List[str],
                               top_n_ic: int = 60,
                               corr_cap: float = 0.90) -> List[str]:
    """Return a pruned feature list for the current window."""
    
    # Compute IC per feature (skip columns with all-NA or constant)
    ic_scores = []
    y = train_df['y_next'].values
    
    for c in candidate_cols:
        if c not in train_df.columns:
            continue
            
        x = train_df[c].values
        if np.all(pd.isna(x)) or np.nanstd(x) < 1e-12:
            continue
            
        # Pairwise valid rows
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
            
        try:
            rho, _ = spearmanr(x[mask], y[mask])
        except Exception:
            rho = np.nan
            
        if np.isfinite(rho):
            ic_scores.append((c, abs(float(rho))))
    
    if not ic_scores:
        return candidate_cols[:top_n_ic]  # fallback
    
    ic_scores.sort(key=lambda t: t[1], reverse=True)
    ranked = [c for c, _ in ic_scores[:top_n_ic]]

    # Greedy correlation pruning
    keep = []
    X = train_df[ranked].fillna(train_df[ranked].median()).values
    col_to_idx = {c: i for i, c in enumerate(ranked)}
    
    for c in ranked:
        ci = col_to_idx[c]
        xi = X[:, ci]
        ok = True
        
        for kept in keep:
            kj = col_to_idx[kept]
            xk = X[:, kj]
            # Pearson corr for redundancy
            r = np.corrcoef(xi, xk)[0, 1]
            if np.isfinite(r) and abs(r) > corr_cap:
                ok = False
                break
        if ok:
            keep.append(c)
    
    return keep

# ---------------------------------------------------------------------
# 5) Main backtest function
# ---------------------------------------------------------------------

def run_backtest(df: pd.DataFrame,
                 train_window: int = 120,
                 cov_lookback: int = 90,
                 topK: int = 15,
                 risk_aversion: float = 5e-3,
                 turnover_penalty: float = 1e-3,
                 w_max: float = 0.20,
                 cost_bps_roundtrip: float = 5.0,
                 min_history: int = 150,
                 use_feature_selection: bool = True,
                 top_n_ic: int = 60,
                 corr_cap: float = 0.90) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the portfolio optimization backtest.
    
    Returns:
        w_df: daily portfolio weights per ticker
        pnl_s: daily realized pnl (after costs)
        turnover_s: daily turnover
    """
    
    print("Starting portfolio optimization backtest...")
    
    # Get unique dates and tickers
    all_dates = sorted(df['date'].unique())
    tickers = sorted(df['ticker'].unique())
    N = len(tickers)
    
    print(f"Universe: {N} tickers, {len(all_dates)} dates")
    
    # Get feature candidates
    feature_candidates = get_feature_candidates(df)
    
    # ML model: Impute -> Scale -> Ridge
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True, random_state=7))
    ])

    # Storage
    w_prev = np.zeros(N)
    w_records = []
    pnl_records = []
    turnover_records = []
    portfolio_values = []
    
    # Initial portfolio value: 1M equally divided among all 45 stocks
    initial_capital = 1_000_000
    initial_weight_per_stock = 1.0 / N
    w_prev = np.full(N, initial_weight_per_stock)
    current_portfolio_value = initial_capital

    # Cost vector per name (per unit absolute change in weight)
    cost_vec = np.full(N, cost_bps_roundtrip/1e4, dtype=float)

    # Precompute panel pivot for quick return matrices
    # Remove duplicates first
    df_unique = df.drop_duplicates(subset=['date', 'ticker']).copy()
    close_pivot = df_unique.pivot(index='date', columns='ticker', values='close').sort_index()
    logret_pivot = np.log(close_pivot/close_pivot.shift(1))

    # Rolling loop: rebalance at t using info up to t-1, hold to t+1
    for t_idx in range(min_history, len(all_dates)-1):
        dt = all_dates[t_idx]
        dt_prev = all_dates[t_idx-1]

        if t_idx % 20 == 0:  # Progress indicator
            print(f"Processing date {dt} ({t_idx}/{len(all_dates)})")

        # TRAIN SET: last train_window days ending t-1 (all tickers)
        train_start_idx = max(0, t_idx - train_window)
        train_dates = all_dates[train_start_idx:t_idx]

        train_mask = df['date'].isin(train_dates)
        train_df = df.loc[train_mask, ['ticker','date','y_next'] + feature_candidates].dropna(subset=['y_next'])

        # Feature selection on the TRAIN window only
        selected_features = feature_candidates
        if use_feature_selection and len(train_df) > 200:
            try:
                selected_features = select_features_time_aware(train_df,
                                                              candidate_cols=feature_candidates,
                                                              top_n_ic=top_n_ic,
                                                              corr_cap=corr_cap)
            except Exception as e:
                print(f"Feature selection failed at {dt}: {e}")
                # Use a subset of features as fallback
                selected_features = feature_candidates[:50]

        X_train = train_df[selected_features].values
        y_train = train_df['y_next'].values
        
        if len(y_train) < 200:
            # Not enough samples; skip trading until we have data
            w_records.append(pd.Series(w_prev, index=tickers, name=dt))
            pnl_records.append(0.0)
            turnover_records.append(0.0)
            continue

        # Train model
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Model training failed at {dt}: {e}")
            w_records.append(pd.Series(w_prev, index=tickers, name=dt))
            pnl_records.append(0.0)
            turnover_records.append(0.0)
            continue

        # PREDICT mu for date t (one-step-ahead for each ticker using features at t)
        x_today = df.loc[df['date'] == dt, ['ticker'] + selected_features].copy()
        if x_today.empty:
            # no data today
            w_records.append(pd.Series(w_prev, index=tickers, name=dt))
            pnl_records.append(0.0)
            turnover_records.append(0.0)
            continue

        mu_pred = pd.Series(0.0, index=tickers)
        try:
            # Ensure we have the same features in the same order as training
            x_today_features = x_today[selected_features].fillna(0.0)  # Fill NaN with 0
            mu_slice = model.predict(x_today_features.values)
            mu_pred.loc[x_today['ticker'].values] = mu_slice
        except Exception as e:
            print(f"Prediction failed at {dt}: {e}")
            mu_pred = pd.Series(0.0, index=tickers)
        
        mu_vec = mu_pred.values

        # COVARIANCE (Ledoit–Wolf) from last cov_lookback days up to t-1
        cov_start_idx = max(0, t_idx - cov_lookback)
        cov_dates = all_dates[cov_start_idx:t_idx]
        R = logret_pivot.loc[cov_dates].dropna(axis=1, how='any').fillna(0.0)

        # Align mu to the covariance tickers; fill missing with 0
        cov_tickers = R.columns.tolist()
        if len(cov_tickers) < 5:
            # if too few names, hold previous
            w_records.append(pd.Series(w_prev, index=tickers, name=dt))
            pnl_records.append(0.0)
            turnover_records.append(0.0)
            continue

        # Build Sigma
        try:
            lw = LedoitWolf().fit(R.values)
            Sigma = lw.covariance_
        except Exception as e:
            print(f"Covariance estimation failed at {dt}: {e}")
            w_records.append(pd.Series(w_prev, index=tickers, name=dt))
            pnl_records.append(0.0)
            turnover_records.append(0.0)
            continue

        # Preselect topK by mu among cov universe
        mu_cov = pd.Series(mu_pred, index=tickers).reindex(cov_tickers).fillna(0.0)
        sel = mu_cov.sort_values(ascending=False).index[:topK].tolist()

        # Map indexes
        idx_map = {tic: i for i, tic in enumerate(tickers)}
        sel_mask_full = np.zeros(len(tickers), dtype=bool)
        for tic in sel:
            if tic in idx_map:
                sel_mask_full[idx_map[tic]] = True

        # Build Sigma_full
        Sigma_full = np.eye(N) * 1e-4
        # Place cov for cov_tickers
        cov_pos = [idx_map[t] for t in cov_tickers if t in idx_map]
        for i, ti in enumerate(cov_pos):
            for j, tj in enumerate(cov_pos):
                Sigma_full[ti, tj] = Sigma[i, j]

        # Prepare optimization vectors
        mu_full = pd.Series(mu_vec, index=tickers).fillna(0.0).values
        w = cp.Variable(N)
        dw = w - w_prev

        objective = (
            mu_full @ w
            - risk_aversion * cp.quad_form(w, Sigma_full)
            - turnover_penalty * cp.norm1(dw)
            - (cost_vec @ cp.abs(dw))
        )
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= w_max,
            w[~sel_mask_full] <= 1e-6,  # effectively exclude non-selected
        ]

        prob = cp.Problem(cp.Maximize(objective), constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7, verbose=False)
            except Exception:
                print(f"Optimization failed at {dt}")
                w_t = np.zeros(N)
                sel_idx = np.where(sel_mask_full)[0]
                if len(sel_idx) > 0:
                    w_t[sel_idx] = 1.0/len(sel_idx)
                w_records.append(pd.Series(w_t, index=tickers, name=dt))
                pnl_records.append(0.0)
                turnover_records.append(0.0)
                continue

        if w.value is None:
            w_t = np.zeros(N)
            # fallback equal-weight in selection
            sel_idx = np.where(sel_mask_full)[0]
            if len(sel_idx) > 0:
                w_t[sel_idx] = 1.0/len(sel_idx)
        else:
            w_t = np.clip(w.value, 0, None)
            s = w_t.sum()
            w_t = w_t / (s + 1e-12)

        # Realized next-day PnL
        next_dt = all_dates[t_idx+1]
        next_ret_full = logret_pivot.loc[next_dt].reindex(tickers).fillna(0.0).values
        gross = float(w_t @ next_ret_full)
        realized_cost = float(np.abs(w_t - w_prev) @ cost_vec)
        pnl = gross - realized_cost
        
        # Update portfolio value
        current_portfolio_value *= (1 + pnl)

        w_records.append(pd.Series(w_t, index=tickers, name=dt))
        pnl_records.append(pnl)
        turnover_records.append(float(np.abs(w_t - w_prev).sum()))
        portfolio_values.append(current_portfolio_value)
        w_prev = w_t

    w_df = pd.DataFrame(w_records)
    pnl_s = pd.Series(pnl_records, index=w_df.index)
    turnover_s = pd.Series(turnover_records, index=w_df.index)
    portfolio_values_s = pd.Series(portfolio_values, index=w_df.index)

    # Performance summary
    cum = (1 + pnl_s).cumprod()
    sharpe = np.sqrt(252) * pnl_s.mean() / (pnl_s.std(ddof=1) + 1e-12)
    max_dd = (cum / cum.cummax() - 1).min()
    total_return = (portfolio_values_s.iloc[-1] / initial_capital - 1)
    
    print("==== Backtest Summary ====")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Final Portfolio Value: ${portfolio_values_s.iloc[-1]:,.0f}")
    print(f"Total Return: {total_return:.1%}")
    print(f"Days: {len(pnl_s)} | Daily Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1%} | Avg Turnover: {turnover_s.mean():.2f}")
    print(f"Volatility: {pnl_s.std() * np.sqrt(252):.1%}")

    return w_df, pnl_s, turnover_s, portfolio_values_s

# ---------------------------------------------------------------------
# 6) Main execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Load data
        df = load_data()
        
        # Add targets
        df = add_targets(df)
        
        # Run backtest
        w_df, pnl_s, turnover_s, portfolio_values_s = run_backtest(df,
            train_window=120,
            cov_lookback=90,
            topK=15,
            risk_aversion=5e-3,
            turnover_penalty=1e-3,
            w_max=0.20,
            cost_bps_roundtrip=5.0,
            min_history=150,
            use_feature_selection=False,  # Disable for now
        )
        
        # Save results
        w_df.to_csv("weights_daily.csv")
        pnl_s.to_csv("pnl_daily.csv")
        turnover_s.to_csv("turnover_daily.csv")
        portfolio_values_s.to_csv("portfolio_values_daily.csv")
        
        print("\nResults saved to:")
        print("- weights_daily.csv")
        print("- pnl_daily.csv") 
        print("- turnover_daily.csv")
        print("- portfolio_values_daily.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
