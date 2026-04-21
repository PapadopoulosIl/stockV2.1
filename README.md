# stockV2.1
import datetime as dt
import sqlite3
import textwrap
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from statsmodels.tsa.stattools import coint

# ── Institutional Engine Modules ─────────────────────────────────────────────
from risk_engine import run_risk_engine
from valuation_engine import run_valuation_engine, generate_valuation_insights

warnings.filterwarnings("ignore")
try:
    pd.options.mode.use_inf_as_na = True  # pandas < 2.0
except Exception:
    pass  # pandas 2.0+ χειρίζεται το inf→NaN αυτόματα στις περισσότερες πράξεις

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]


def wrap_text(text, width=120):
    return "\n".join(textwrap.fill(line, width=width) for line in str(text).splitlines())


def calculate_rsi(price_series, period=14):
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(com=period - 1, adjust=False).mean() / down.ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def safe_last(series, default=np.nan):
    clean = series.dropna()
    return clean.iloc[-1] if not clean.empty else default


def safe_get_attr(obj, attr_name, default=None):
    try:
        value = getattr(obj, attr_name)
        return default if value is None else value
    except Exception:
        return default


def format_human_value(value):
    if pd.isna(value):
        return "-"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def format_percent(value):
    return "-" if pd.isna(value) else f"{float(value):+.2f}%"


def format_ratio(value):
    return "-" if pd.isna(value) else f"{float(value):.2f}x"


def metric_delta_text(stock_value, reference_value):
    if pd.isna(stock_value) or pd.isna(reference_value):
        return "n/a"
    diff = float(stock_value) - float(reference_value)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f}"


def determine_period_config(range_key):
    configs = {
        "1D": {"period": "1d", "interval": "5m", "label": "Today + Pre / After Hours"},
        "1W": {"period": "5d", "interval": "30m", "label": "1 Week Pulse"},
        "1M": {"period": "1mo", "interval": "1h", "label": "1 Month Pulse"},
        "1Y": {"period": "1y", "interval": "1d", "label": "1 Year Pulse"},
    }
    return configs[range_key]


# ---------------------------------------------------------------------------
# Portfolio — Institutional Backend (Transactions Ledger Architecture)
# ---------------------------------------------------------------------------
DB_PATH = "local_portfolio.db"
# Metadata expires after 24h and gets refreshed on next load
METADATA_TTL_HOURS = 24


def _db_conn():
    return sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)


def init_portfolio_db():
    """Schema migration-safe: adds tables/columns without wiping existing data."""
    with _db_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS transactions
               (id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                action          TEXT NOT NULL CHECK(action IN ('BUY','SELL')),
                shares          REAL NOT NULL CHECK(shares > 0),
                execution_price REAL NOT NULL CHECK(execution_price > 0),
                timestamp       TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS asset_metadata
               (ticker       TEXT PRIMARY KEY,
                sector       TEXT,
                industry     TEXT,
                asset_class  TEXT,
                beta         REAL,
                div_yield    REAL,
                last_updated TEXT)"""
        )
        
        # Schema Migrations for advanced metrics
        for col in ["trailing_pe", "forward_pe", "price_to_sales", "projected_growth"]:
            try:
                conn.execute(f"ALTER TABLE asset_metadata ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
        # Index for fast per-user queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_user ON transactions(username, ticker)"
        )


# ── DB helpers ──────────────────────────────────────────────────────────────

def db_add_user(username: str) -> None:
    with _db_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))


def db_log_transaction(username: str, ticker: str, action: str,
                        shares: float, execution_price: float) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO transactions
               (username, ticker, action, shares, execution_price, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (username, ticker.upper(), action, shares, execution_price, ts),
        )


def db_get_transactions(username: str) -> pd.DataFrame:
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM transactions WHERE username=? ORDER BY timestamp ASC",
            conn, params=(username,),
        )


def db_delete_transaction(tx_id: int) -> None:
    with _db_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE id=?", (tx_id,))


def db_upsert_metadata(ticker: str, info: dict) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO asset_metadata
               (ticker, sector, industry, asset_class, beta, div_yield, trailing_pe, forward_pe, price_to_sales, projected_growth, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker) DO UPDATE SET
                 sector=excluded.sector, industry=excluded.industry,
                 asset_class=excluded.asset_class,
                 beta=excluded.beta, div_yield=excluded.div_yield,
                 trailing_pe=excluded.trailing_pe, forward_pe=excluded.forward_pe,
                 price_to_sales=excluded.price_to_sales, projected_growth=excluded.projected_growth,
                 last_updated=excluded.last_updated""",
            (
                ticker.upper(),
                info.get("sector") or "Unknown",
                info.get("industry") or "Unknown",
                info.get("quoteType") or "EQUITY",
                _safe_float(info.get("beta")),
                _safe_float(info.get("dividendYield")),
                _safe_float(info.get("trailingPE")),
                _safe_float(info.get("forwardPE")),
                _safe_float(info.get("priceToSalesTrailing12Months")),
                _safe_float(info.get("earningsGrowth")),
                now,
            ),
        )


def db_get_metadata(tickers: list) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    with _db_conn() as conn:
        return pd.read_sql_query(
            f"SELECT * FROM asset_metadata WHERE ticker IN ({placeholders})",
            conn, params=tickers,
        )


def _safe_float(v) -> float:
    """Coerce a value to float, return NaN if impossible."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ── Metadata enrichment (with TTL refresh) ──────────────────────────────────

def enrich_metadata(tickers: list) -> None:
    """
    Fetches metadata for:
      - tickers not yet in the DB
      - tickers whose last_updated is older than METADATA_TTL_HOURS
    Runs synchronously but is gated so it only fires when needed.
    """
    if not tickers:
        return

    existing = db_get_metadata(tickers)
    now = dt.datetime.now()
    stale_or_missing = []

    for t in tickers:
        if existing.empty or t not in existing["ticker"].values:
            stale_or_missing.append(t)
            continue
        row = existing[existing["ticker"] == t].iloc[0]
        
        # Check for newly added schema fields that may be null
        if "forward_pe" not in row or pd.isna(row["forward_pe"]):
            stale_or_missing.append(t)
            continue
            
        try:
            last = dt.datetime.fromisoformat(row["last_updated"])
            if (now - last).total_seconds() > METADATA_TTL_HOURS * 3600:
                stale_or_missing.append(t)
        except (TypeError, ValueError):
            stale_or_missing.append(t)

    if not stale_or_missing:
        return

    # Show a lightweight spinner only when we actually need to fetch
    with st.spinner(f"Fetching metadata for {', '.join(stale_or_missing)}…"):
        for t in stale_or_missing:
            try:
                info = yf.Ticker(t).info or {}
                # Ensure ETFs get their multiples populated via fallback
                fun = fetch_fundamentals_data(t).get("snapshot", {})
                if pd.notna(fun.get("forward_pe")) and pd.isna(info.get("forwardPE")): 
                    info["forwardPE"] = fun["forward_pe"]
                if pd.notna(fun.get("trailing_pe")) and pd.isna(info.get("trailingPE")): 
                    info["trailingPE"] = fun["trailing_pe"]
                    
                db_upsert_metadata(t, info)
            except Exception:
                # Write a stub so we don't retry on every load for bad tickers
                db_upsert_metadata(t, {})


# ── Market snapshot (current price + prev close, weekend-safe) ───────────────

def fetch_market_snapshot(tickers: list) -> dict:
    """
    Returns {ticker: {"current": float, "prev_close": float, "spark": list[float]}}.
    period="7d" gives ~5 trading sessions for the sparkline + prev_close fallback.
    Weekend-safe: always enough rows even over holidays.
    """
    result: dict = {}
    if not tickers:
        return result
    try:
        raw = yf.download(
            tickers, period="7d", interval="1d",
            progress=False, auto_adjust=False,
        )
        if raw.empty:
            return result

        # Normalise to DataFrame with ticker columns
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)

        for t in tickers:
            col = t if t in close.columns else None
            if col is None:
                continue
            s = close[col].dropna()
            if s.empty:
                continue
            spark = [round(float(v), 4) for v in s.values]
            result[t] = {
                "current":    float(s.iloc[-1]),
                "prev_close": float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1]),
                "spark":      spark,           # full 7-day close list for LineChartColumn
            }
    except Exception:
        pass
    return result
    full_target = list(set(tickers + ["SPY"]))
    try:
        raw = yf.download(
            full_target, period="7d", interval="1d",
            progress=False, auto_adjust=False,
        )
        if raw.empty:
            return result

        # Normalise to DataFrame with ticker columns
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=full_target[0])
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)

        for t in full_target:
            col = t if t in close.columns else None
            if col is None:
                continue
            s = close[col].dropna()
            if s.empty:
                continue
            spark = [round(float(v), 4) for v in s.values]
            result[t] = {
                "current":    float(s.iloc[-1]),
                "prev_close": float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1]),
                "spark":      spark,           # full 7-day close list
            }
    except Exception:
        pass
    return result


# ── Core portfolio computation (FIFO + DCA, fully vectorised per ticker) ─────

def _compute_open_lots(grp: pd.DataFrame):
    """
    FIFO lot matching for a single ticker's transaction history.
    Returns (buy_queue, realized_pl) where buy_queue is a list of
    [shares_remaining, price, timestamp_str].
    """
    buy_queue: list = []
    realized_pl = 0.0

    for row in grp.itertuples(index=False):
        if row.action == "BUY":
            buy_queue.append([float(row.shares), float(row.execution_price), row.timestamp])
        elif row.action == "SELL":
            remaining = float(row.shares)
            sell_px   = float(row.execution_price)
            for lot in buy_queue:
                if remaining <= 0:
                    break
                consumed = min(lot[0], remaining)
                realized_pl += consumed * (sell_px - lot[1])
                lot[0]      -= consumed
                remaining   -= consumed
            buy_queue = [lot for lot in buy_queue if lot[0] > 1e-9]

    return buy_queue, realized_pl


def compute_portfolio_state(txns: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    """
    Aggregates the full transaction ledger into one row per open ticker.
    Columns returned: Ticker, Shares, Avg Cost, Current Price, Cost Basis,
    Current Value, Unrealized P/L ($), Unrealized P/L (%),
    Day's Gain ($), Day's Gain (%), Realized P/L ($).
    """
    if txns.empty:
        return pd.DataFrame()

    today_str = dt.date.today().isoformat()
    rows = []

    for ticker, grp in txns.groupby("ticker", sort=False):
        buy_queue, realized_pl = _compute_open_lots(grp)

        total_shares = sum(lot[0] for lot in buy_queue)
        if total_shares < 1e-9:
            continue  # fully closed position

        total_cost = sum(lot[0] * lot[1] for lot in buy_queue)
        avg_cost   = total_cost / total_shares
        cost_basis = total_cost   # = total_shares * avg_cost

        mkt          = market_data.get(ticker, {})
        current_px   = mkt.get("current",    np.nan)
        prev_close   = mkt.get("prev_close", np.nan)

        current_value  = total_shares * current_px  if pd.notna(current_px)  else np.nan
        unrealized_pl  = current_value - cost_basis if pd.notna(current_value) else np.nan
        unrealized_pct = (unrealized_pl / cost_basis * 100) if (pd.notna(unrealized_pl) and cost_basis) else np.nan

        # Day's Gain anchor:
        #   If every remaining open lot was bought today → use avg_cost
        #   Otherwise → use yesterday's close (standard brokerage behaviour)
        all_bought_today = all(lot[2][:10] == today_str for lot in buy_queue)
        day_anchor  = avg_cost if all_bought_today else prev_close
        days_gain   = (current_px - day_anchor) * total_shares if pd.notna(current_px) and pd.notna(day_anchor) else np.nan
        days_gain_pct = (days_gain / (day_anchor * total_shares) * 100) if (pd.notna(days_gain) and day_anchor and day_anchor * total_shares != 0) else np.nan

        rows.append({
            "Ticker":           ticker,
            "Shares":           round(total_shares, 6),
            "Avg Cost":         round(avg_cost,     4),
            "Current Price":    round(current_px,   4) if pd.notna(current_px)   else np.nan,
            "Cost Basis":       round(cost_basis,   2),
            "Current Value":    round(current_value, 2) if pd.notna(current_value) else np.nan,
            "Unrealized P/L ($)":  round(unrealized_pl,  2) if pd.notna(unrealized_pl)  else np.nan,
            "Unrealized P/L (%)":  round(unrealized_pct, 2) if pd.notna(unrealized_pct) else np.nan,
            "Day's Gain ($)":      round(days_gain,      2) if pd.notna(days_gain)      else np.nan,
            "Day's Gain (%)":      round(days_gain_pct,  2) if pd.notna(days_gain_pct)  else np.nan,
            "Realized P/L ($)":    round(realized_pl,    2),
        })

    return pd.DataFrame(rows)


# ── Validation helper ────────────────────────────────────────────────────────

def validate_sell(txns: pd.DataFrame, ticker: str, sell_shares: float) -> tuple[bool, str]:
    """Returns (ok, error_message). Checks you can't sell more than you hold."""
    if txns.empty:
        return False, "Δεν υπάρχουν συναλλαγές για αυτό το ticker."
    t_txns = txns[txns["ticker"] == ticker.upper()]
    if t_txns.empty:
        return False, f"Δεν κατέχεις {ticker}."
    buy_queue, _ = _compute_open_lots(t_txns)
    held = sum(lot[0] for lot in buy_queue)
    if sell_shares > held + 1e-9:
        return False, f"Προσπαθείς να πουλήσεις {sell_shares:.4f} μετοχές αλλά κατέχεις μόνο {held:.4f}."
    return True, ""


# ── Weighted portfolio-level risk metrics ────────────────────────────────────

def compute_weighted_metrics(port_df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """
    Returns {"beta": float, "div_yield_pct": float} using value-weights.
    Excludes positions without valid market data.
    """
    out = {"beta": np.nan, "div_yield_pct": np.nan}
    if meta.empty or port_df.empty:
        return out

    meta_idx = meta.set_index("ticker")
    valid = port_df.dropna(subset=["Current Value"]).copy()
    if valid.empty:
        return out

    total_val = valid["Current Value"].sum()
    if total_val <= 0:
        return out

    valid["_w"] = valid["Current Value"] / total_val
    valid["_beta"] = valid["Ticker"].map(
        lambda t: _safe_float(meta_idx.loc[t, "beta"]) if t in meta_idx.index else np.nan
    )
    valid["_div"] = valid["Ticker"].map(
        lambda t: _safe_float(meta_idx.loc[t, "div_yield"]) if t in meta_idx.index else np.nan
    )

    beta_valid = valid.dropna(subset=["_beta"])
    if not beta_valid.empty:
        w = beta_valid["_w"] / beta_valid["_w"].sum()   # re-normalise for missing tickers
        out["beta"] = float((beta_valid["_beta"] * w).sum())

    div_valid = valid.dropna(subset=["_div"])
    if not div_valid.empty:
        w = div_valid["_w"] / div_valid["_w"].sum()
        out["div_yield_pct"] = float((div_valid["_div"] * w).sum() * 100)

    return out

# ── Institutional Risk & Advisory System (Quant & Valuation Engine) ──────────

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_historical_returns(tickers: list) -> pd.DataFrame:
    """Fetches 1y of daily returns for risk logic."""
    if not tickers:
        return pd.DataFrame()
    
    full_list = list(set(tickers + ["SPY", "QQQ"]))
    try:
        raw = yf.download(full_list, period="1y", interval="1d", progress=False, auto_adjust=False)
        if raw.empty: return pd.DataFrame()
        
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=full_list[0])
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)
            
        close.ffill(inplace=True)
        returns = close.pct_change().dropna()
        return returns
    except Exception:
        return pd.DataFrame()

# ── Legacy wrappers — τώρα delegate στα engine modules ──────────────────────

def calculate_quant_metrics(port_df: pd.DataFrame, returns_df: pd.DataFrame,
                             sector_map: dict | None = None) -> dict:
    """
    Wrapper που καλεί το risk_engine.run_risk_engine().
    Διατηρεί backwards-compatible keys + προσθέτει τα νέα institutional metrics.
    """
    result = run_risk_engine(port_df, returns_df, sector_map=sector_map)
    # Backwards compat alias
    result["weighted_forward_pe"] = np.nan  # filled by valuation engine
    return result


def calculate_valuation_metrics(port_df: pd.DataFrame, meta_df: pd.DataFrame,
                                  risk_free_rate: float = 0.043) -> dict:
    """
    Wrapper που καλεί το valuation_engine.run_valuation_engine().
    Backwards-compatible: εκθέτει 'weighted_forward_pe' = harmonic P/E.
    """
    result = run_valuation_engine(port_df, meta_df, risk_free_rate=risk_free_rate)
    # Backwards compat: το παλιό key ήταν 'weighted_forward_pe'
    result["weighted_forward_pe"] = result.get("harmonic_forward_pe", np.nan)
    return result


def generate_advisory_insights(quant: dict, valuation: dict, total_value: float) -> list[str]:
    """
    Συνδυάζει insights από Risk Engine + Valuation Engine.
    Επιστρέφει unified list[str] για το render_advisory_tab.
    """
    insights = []
    
    # ── Market Regime Context ───────────────────────────────────────────────
    regime = quant.get("market_regime", "UNKNOWN")
    regime_texts = {
        "BEAR_PANIC": "Regime Warning (Bear / High Volatility): Η αγορά βρίσκεται κάτω από τον ΚΜΟ 200 με αυξημένη μεταβλητότητα. Προτεραιότητα η διατήρηση κεφαλαίου.",
        "BEAR_GRINDING": "Regime Setup (Bear / Low Volatility): Γραμμική πτώση της αγοράς. Αναζητήστε Value factors και defensive margins.",
        "BULL_VOLATILE": "Regime Setup (Bull / High Volatility): Ανοδική πορεία αλλά με αναταράξεις στο tape. Tighter stop-losses συνιστώνται.",
        "BULL_STEADY": "Regime OK (Bull Steady): Η αγορά επιβεβαιώνει σταθερό uptrend. Tactical συμμετοχή σε Growth assets."
    }
    if regime in regime_texts:
        insights.append(f"**{regime_texts[regime]}**")

    # ── Risk Insights ────────────────────────────────────────────────────────
    var_abs = quant.get("var_95", np.nan)
    cvar_abs = quant.get("cvar_95", np.nan)
    var_pct = quant.get("var_95_pct", np.nan)
    cvar_pct = quant.get("cvar_95_pct", np.nan)
    vol_annual = quant.get("portfolio_vol_annual", np.nan)

    # VaR + CVaR combined insight
    if pd.notna(var_abs) and pd.notna(cvar_abs) and total_value > 0:
        if var_pct > 3.0:
            insights.append(
                f"**High Tail Risk:** VaR (95%) = **-{var_pct:.1f}%** (${var_abs:,.0f}). "
                f"Expected Shortfall (CVaR) = **-{cvar_pct:.1f}%** (${cvar_abs:,.0f}) (Μέση ζημιά στο worst 5%). "
                f"Εξετάστε hedging ή μείωση concentrated positions."
            )
        else:
            insights.append(
                f"**Stable Risk Profile:** VaR (95%) = -{var_pct:.1f}% / CVaR = -{cvar_pct:.1f}%. "
                f"Επιθυμητό risk profile. Annualised Portfolio Volatility: {vol_annual:.1f}%."
            )

    # Rolling Correlation Alert
    if quant.get("rolling_corr_alert", False):
        if regime in ["BEAR_PANIC", "BEAR_GRINDING"]:
            insights.append(
                "**Correlation Regime Shift:** Οι 30-Day συσχετίσεις αυξήθηκαν ραγδαία. Δεδομένου του αρνητικού market regime, ελλοχεύει σοβαρός κίνδυνος ρευστοποιήσεων (liquidity trap). Μειώστε συσχετισμένα (correlated) assets."
            )
        else:
            insights.append(
                "**Correlation Regime Shift:** Οι 30-Day συσχετίσεις αυξήθηκαν. Η διαφοροποίηση του risk factor (diversification) μειώνεται, κάτι που σημαίνει υψηλότερο draw-down αν η αγορά αλλάξει κατεύθυνση."
            )

    # Static Correlation Warnings
    corrs = quant.get("correlation_warnings", [])
    if corrs:
        top_corrs = [f"{c[0]}-{c[1]} ({c[2]:.2f})" for c in corrs[:3]]
        insights.append(
            f"**Concentration Warning:** Υψηλή στατική συσχέτιση: {', '.join(top_corrs)}. "
            f"Synchronized drawdown risk — αν πιιεστεί το ένα, πιέζονται όλα."
        )
    else:
        if quant.get("var_95") is not np.nan:  # only if engine ran
            insights.append(
                "**Diversification Confirmed:** Δεν βρέθηκαν στατικές συσχετίσεις >0.80. "
                "Το systematic allocation είναι ισορροπημένο."
            )

    # ── Valuation Insights ───────────────────────────────────────────────────
    val_insights = generate_valuation_insights(valuation, total_value)
    # Strip emojis from the generated valuation insights
    val_insights = [text.replace("🔴 ", "").replace("✅ ", "") for text in val_insights]
    insights.extend(val_insights)

    # ── Stress Test Summary (top 2 negative scenarios) ───────────────────────
    stress = quant.get("stress_results", {})
    if stress:
        negative = sorted(
            [(k, v) for k, v in stress.items() if v["pct"] < 0],
            key=lambda x: x[1]["pct"]
        )[:2]
        if negative:
            stress_lines = [f"{k}: ${v['pnl']:,.0f} ({v['pct']:.1f}%)" for k, v in negative]
            insights.append(
                "**Stress Test Alerts (Worst Case):** " + " | ".join(stress_lines)
            )

    return insights

# ── Render ───────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return (
        f'<div class="glass-card">'
        f'<div class="kpi-title">{label}</div>'
        f'<div class="kpi-value" style="font-size:1.05rem;">{value}</div>'
        f'{delta_html}</div>'
    )


def render_portfolio_tab() -> None:
    # ── Login gate ──────────────────────────────────────────────────────────
    if "logged_in_user" not in st.session_state:
        col_form, _ = st.columns([1, 2])
        with col_form:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔐 Είσοδος")
            username = st.text_input("Username:", key="login_input")
            if st.button("Σύνδεση", use_container_width=True):
                u = username.strip()
                if u:
                    db_add_user(u)
                    st.session_state.logged_in_user = u
                    st.rerun()
                else:
                    st.warning("Εισάγετε ένα έγκυρο username.")
            st.markdown("</div>", unsafe_allow_html=True)
        return

    user = st.session_state.logged_in_user
    hdr_l, hdr_r = st.columns([5, 1])
    hdr_l.markdown(
        f'<span style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;">Portfolio Wealth Terminal</span>'
        f'&nbsp;&nbsp;<span style="font-weight:700;color:var(--text);">{user}</span>',
        unsafe_allow_html=True,
    )
    if hdr_r.button("Sign out", use_container_width=True):
        del st.session_state.logged_in_user
        st.rerun()

    # ── Transaction entry ────────────────────────────────────────────────────
    with st.expander("＋ New Transaction", expanded=False):
        with st.form("tx_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            tx_ticker = c1.text_input("Ticker").strip().upper()
            tx_action = c2.selectbox("Side", ["BUY", "SELL"])
            tx_shares = c3.number_input("Qty", min_value=0.0001, step=1.0, format="%.4f")
            tx_price  = c4.number_input("Execution Price ($)", min_value=0.0001, step=0.01, format="%.4f")
            submitted = st.form_submit_button("Submit")

        if submitted:
            if not tx_ticker or tx_shares <= 0 or tx_price <= 0:
                st.error("All fields required.")
            elif tx_action == "SELL":
                current_txns = db_get_transactions(user)
                ok, msg = validate_sell(current_txns, tx_ticker, tx_shares)
                if not ok:
                    st.error(f"⚠️ {msg}")
                else:
                    db_log_transaction(user, tx_ticker, tx_action, tx_shares, tx_price)
                    st.success(f"SELL {tx_shares:.4f} {tx_ticker} @ ${tx_price:.2f} booked.")
                    st.rerun()
            else:
                db_log_transaction(user, tx_ticker, tx_action, tx_shares, tx_price)
                st.success(f"BUY {tx_shares:.4f} {tx_ticker} @ ${tx_price:.2f} booked.")
                st.rerun()

    # ── Load & compute ───────────────────────────────────────────────────────
    txns = db_get_transactions(user)
    if txns.empty:
        st.info("No transactions yet. Add your first position above.")
        return

    unique_tickers = txns["ticker"].unique().tolist()
    enrich_metadata(unique_tickers)
    meta_df     = db_get_metadata(unique_tickers)
    market_data = fetch_market_snapshot(unique_tickers)
    port_df     = compute_portfolio_state(txns, market_data)

    if port_df.empty:
        st.info("No open positions — all lots have been closed.")
        return

    meta_idx = meta_df.set_index("ticker") if not meta_df.empty else pd.DataFrame()

    port_df["Sector"] = port_df["Ticker"].map(
        lambda t: str(meta_idx.loc[t, "sector"])
        if (not meta_idx.empty and t in meta_idx.index) else "Unknown"
    )
    port_df["7D Trend"] = port_df["Ticker"].map(
        lambda t: market_data.get(t, {}).get("spark", [])
    )

    total_val        = port_df["Current Value"].dropna().sum()
    port_df["Wt %"]  = (port_df["Current Value"] / total_val * 100).round(2) if total_val else np.nan

    # ── Aggregates ───────────────────────────────────────────────────────────
    total_cost       = port_df["Cost Basis"].sum()
    total_mkt        = port_df["Current Value"].sum()
    total_unreal     = port_df["Unrealized P/L ($)"].sum()
    total_unreal_pct = (total_unreal / total_cost * 100) if total_cost else 0.0
    total_days_gain  = port_df["Day's Gain ($)"].sum()
    total_realized   = port_df["Realized P/L ($)"].sum()

    risk      = compute_weighted_metrics(port_df, meta_df)
    beta_str  = f"{risk['beta']:.2f}"           if pd.notna(risk["beta"])          else "—"
    yield_str = f"{risk['div_yield_pct']:.2f}%" if pd.notna(risk["div_yield_pct"]) else "—"

    # ── Top KPI bar ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    unreal_colour = "var(--green)" if total_unreal >= 0 else "var(--red)"
    dg_colour     = "var(--green)" if total_days_gain >= 0 else "var(--red)"

    for col, label, value, colour in [
        (k1, "Cost Basis",     f"${total_cost:,.2f}",          "var(--text)"),
        (k2, "Market Value",   f"${total_mkt:,.2f}",           "var(--text)"),
        (k3, "Total P/L",      f"${total_unreal:+,.2f} ({total_unreal_pct:+.1f}%)", unreal_colour),
        (k4, "Day's Gain",     f"${total_days_gain:+,.2f}",    dg_colour),
        (k5, "Portfolio Beta", beta_str,                        "var(--text)"),
        (k6, "Wtd Div Yield",  yield_str,                       "var(--text)"),
    ]:
        col.markdown(
            f'<div class="glass-card">'
            f'<div class="kpi-title">{label}</div>'
            f'<div class="kpi-value" style="font-size:.95rem;color:{colour};">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Realized P/L inline pill (only when non-zero)
    if abs(total_realized) > 0.005:
        rc = "var(--green)" if total_realized >= 0 else "var(--red)"
        st.markdown(
            f'<div style="padding:.3rem .8rem;border-radius:4px;border:1px solid {rc}33;'
            f'background:{rc}11;color:{rc};font-size:.78rem;font-weight:600;display:inline-block;margin-bottom:.5rem;">'
            f'Realized P/L (FIFO closed lots): ${total_realized:+,.2f}</div>',
            unsafe_allow_html=True,
        )

    # ── Main layout: 25 % allocation panel │ 75 % ledger ────────────────────
    left_col, right_col = st.columns([1, 3], gap="medium")

    # ── LEFT: Donut + Sector bar ─────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="section-label">Allocation</div>', unsafe_allow_html=True)

        donut_src = port_df[port_df["Current Value"].notna() & (port_df["Current Value"] > 0)][
            ["Ticker", "Current Value", "Sector"]
        ].copy()

        if not donut_src.empty:
            donut_src["Pct"] = (donut_src["Current Value"] / donut_src["Current Value"].sum() * 100).round(1)

            donut = (
                alt.Chart(donut_src)
                .mark_arc(innerRadius=60, stroke="#131722", strokeWidth=1.5)
                .encode(
                    theta=alt.Theta("Current Value:Q", stack=True),
                    color=alt.Color(
                        "Ticker:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(
                            labelColor="#d1d4dc", title=None,
                            orient="bottom", columns=2,
                            labelFontSize=11,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Ticker:N", title="Ticker"),
                        alt.Tooltip("Current Value:Q", format="$,.2f", title="Value"),
                        alt.Tooltip("Pct:Q", format=".1f", title="Wt %"),
                        alt.Tooltip("Sector:N", title="Sector"),
                    ],
                )
                .properties(height=260)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(donut, use_container_width=True)

            # Sector exposure horizontal bar
            sector_src = (
                donut_src.groupby("Sector")["Current Value"]
                .sum().reset_index()
                .rename(columns={"Current Value": "Val"})
            )
            sector_src["Wt %"] = (sector_src["Val"] / sector_src["Val"].sum() * 100).round(1)

            sector_bar = (
                alt.Chart(sector_src)
                .mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2, color="#2962ff")
                .encode(
                    x=alt.X("Wt %:Q", axis=alt.Axis(labelColor="#787b86", title=None, grid=False, labelFontSize=10)),
                    y=alt.Y("Sector:N", sort="-x", axis=alt.Axis(labelColor="#d1d4dc", title=None, labelFontSize=10)),
                    tooltip=["Sector:N", alt.Tooltip("Wt %:Q", format=".1f", title="Weight %")],
                )
                .properties(
                    height=max(60, len(sector_src) * 28),
                    title=alt.Title("Sector Exposure", color="#787b86", fontSize=10),
                )
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(sector_bar, use_container_width=True)

        # ── Portfolio totals summary ─────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:.5rem;">Summary</div>', unsafe_allow_html=True)
        for label, val in [
            ("Positions",   str(len(port_df))),
            ("Cost Basis",  f"${total_cost:,.2f}"),
            ("Mkt Value",   f"${total_mkt:,.2f}"),
            ("Open P/L",    f"${total_unreal:+,.2f}"),
            ("Beta",        beta_str),
            ("Div Yield",   yield_str),
        ]:
            colour_val = ""
            if label == "Open P/L":
                colour_val = "color:var(--green);" if total_unreal >= 0 else "color:var(--red);"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:.2rem 0;'
                f'border-bottom:1px solid var(--border);font-size:.8rem;">'
                f'<span style="color:var(--muted);">{label}</span>'
                f'<span style="font-weight:600;{colour_val}">{val}</span></div>',
                unsafe_allow_html=True,
            )

    # ── RIGHT: Position ledger with sparklines ───────────────────────────────
    with right_col:
        st.markdown('<div class="section-label">Open Positions</div>', unsafe_allow_html=True)

        # Build two DataFrames:
        # (a) fmt_df  — all text columns, formatted with .style
        # (b) spark_df — only the sparkline column (list of floats)
        # We pass them together via column_config

        def _pl_colour(val):
            if not isinstance(val, (int, float)) or np.isnan(val):
                return ""
            return "color: #26a69a; font-weight:600;" if val >= 0 else "color: #ef5350; font-weight:600;"

        ledger_df = port_df[[
            "Ticker", "Sector", "Shares", "Avg Cost",
            "Current Price", "Current Value", "Wt %",
            "Unrealized P/L ($)", "Unrealized P/L (%)",
            "Day's Gain ($)", "Day's Gain (%)",
            "7D Trend",
        ]].copy()

        # Rename to institutional terminology
        ledger_df = ledger_df.rename(columns={
            "Shares":             "Qty",
            "Avg Cost":           "Cost Basis",
            "Current Price":      "Last",
            "Current Value":      "Mkt Value",
            "Wt %":               "Wt %",
            "Unrealized P/L ($)": "Open P/L ($)",
            "Unrealized P/L (%)": "Open P/L (%)",
            "Day's Gain ($)":     "Day P/L ($)",
            "Day's Gain (%)":     "Day P/L (%)",
        })

        fmt_cols = {
            "Qty":          "{:.4f}",
            "Cost Basis":   "${:.2f}",
            "Last":         "${:.2f}",
            "Mkt Value":    "${:,.2f}",
            "Wt %":         "{:.2f}%",
            "Open P/L ($)": "${:+,.2f}",
            "Open P/L (%)": "{:+.2f}%",
            "Day P/L ($)":  "${:+,.2f}",
            "Day P/L (%)":  "{:+.2f}%",
        }
        pl_cols = ["Open P/L ($)", "Open P/L (%)", "Day P/L ($)", "Day P/L (%)"]

        styler = (
            ledger_df.drop(columns=["7D Trend"])   # keep spark out of .style
            .style
            .format(fmt_cols)
        )
        
        if hasattr(styler, "map"):
            styled = styler.map(_pl_colour, subset=pl_cols)
        else:
            styled = styler.applymap(_pl_colour, subset=pl_cols)

        st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
            height=max(200, len(ledger_df) * 38 + 48),
            column_config={
                # Reintroduce sparkline column via column_config
                # We pass the raw list column from the original df
            },
        )

        # Sparkline table rendered separately with st.dataframe + column_config
        # (Streamlit requires the list column to be in the df passed to dataframe)
        spark_only = ledger_df[["Ticker", "7D Trend"]].copy()
        spark_only = spark_only[spark_only["7D Trend"].apply(lambda x: isinstance(x, list) and len(x) > 1)]

        if not spark_only.empty:
            st.markdown(
                '<div class="section-label" style="margin-top:.8rem;">7-Day Price Trend</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                spark_only,
                hide_index=True,
                use_container_width=True,
                height=max(100, len(spark_only) * 52 + 48),
                column_config={
                    "Ticker":   st.column_config.TextColumn("Ticker", width="small"),
                    "7D Trend": st.column_config.LineChartColumn(
                        "7-Day Trend",
                        width="large",
                        y_min=None,
                        y_max=None,
                    ),
                },
            )

    # ── Transaction ledger (collapsible) ────────────────────────────────────
    with st.expander("📋 Transaction Ledger", expanded=False):
        tx_label_map: dict = {
            int(row.id): (
                f"#{int(row.id):>4} │ {row.action:<4} │ {row.ticker:<6} │ "
                f"{float(row.shares):.4f} @ ${float(row.execution_price):.2f} │ {row.timestamp}"
            )
            for row in txns.itertuples(index=False)
        }
        tx_display = txns[["id", "ticker", "action", "shares", "execution_price", "timestamp"]].copy()
        tx_display.columns = ["ID", "Ticker", "Side", "Qty", "Price ($)", "Timestamp"]
        st.dataframe(
            tx_display.style.format({"Qty": "{:.4f}", "Price ($)": "${:.2f}"}),
            hide_index=True, use_container_width=True, height=220,
        )
        st.markdown("---")
        st.caption("⚠️ Deletion is irreversible and recalculates FIFO P/L.")
        del_id = st.selectbox(
            "Select transaction to delete:",
            options=list(tx_label_map.keys()),
            format_func=lambda x: tx_label_map[x],
        )
        if st.button("Delete Entry", type="primary"):
            db_delete_transaction(del_id)
            st.toast(f"Transaction #{del_id} deleted.")
            st.rerun()


def format_statement_period(column):
    return pd.to_datetime(column).strftime("%b %Y")


def pick_first_available(series, candidates):
    for candidate in candidates:
        if candidate in series.index and pd.notna(series[candidate]):
            return series[candidate]
    return np.nan


def build_statement_table(statement, row_map, periods=4):
    if statement is None or statement.empty:
        return pd.DataFrame()
    working = statement.copy()
    working = working.loc[:, ~working.columns.duplicated()]
    working = working.iloc[:, :periods]

    rows = {}
    for label, candidates in row_map.items():
        for candidate in candidates:
            if candidate in working.index:
                rows[label] = working.loc[candidate]
                break

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).T
    table.columns = [format_statement_period(col) for col in table.columns]
    return table


def add_margin_rows(table):
    if table.empty or "Revenue" not in table.index:
        return table
    revenue = pd.to_numeric(table.loc["Revenue"], errors="coerce").replace(0, np.nan)
    if "Gross Profit" in table.index:
        gross_profit = pd.to_numeric(table.loc["Gross Profit"], errors="coerce")
        table.loc["Profit Margin %"] = gross_profit / revenue * 100
    if "Net Income" in table.index:
        net_income = pd.to_numeric(table.loc["Net Income"], errors="coerce")
        table.loc["Net Profit Margin %"] = net_income / revenue * 100
    return table


def normalize_estimate_table(table):
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return pd.DataFrame()

    normalized = table.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [" ".join(map(str, col)).strip() for col in normalized.columns]

    if "avg" in normalized.index or "Avg" in normalized.index:
        normalized = normalized.T

    normalized.index = [str(idx) for idx in normalized.index]
    rename_map = {
        "numberOfAnalysts": "Analysts",
        "yearAgoRevenue": "Year Ago Revenue",
        "yearAgoEps": "Year Ago EPS",
        "quarterAgoEps": "Quarter Ago EPS",
        "current": "Current",
        "7daysAgo": "7 Days Ago",
        "30daysAgo": "30 Days Ago",
        "60daysAgo": "60 Days Ago",
        "90daysAgo": "90 Days Ago",
        "avg": "Avg",
        "low": "Low",
        "high": "High",
        "growth": "Growth %",
    }
    normalized = normalized.rename(columns=rename_map)
    preferred = [
        col
        for col in [
            "Avg",
            "Low",
            "High",
            "Growth %",
            "Analysts",
            "Year Ago Revenue",
            "Year Ago EPS",
            "Quarter Ago EPS",
            "Current",
            "7 Days Ago",
            "30 Days Ago",
            "60 Days Ago",
            "90 Days Ago",
        ]
        if col in normalized.columns
    ]
    remaining = [col for col in normalized.columns if col not in preferred]
    return normalized[preferred + remaining]


def extract_fund_equity_ratio(funds_data, metric_name, symbol):
    try:
        equity_holdings = funds_data.equity_holdings
        if equity_holdings is None or equity_holdings.empty:
            return np.nan
        if metric_name in equity_holdings.index and symbol in equity_holdings.columns:
            value = equity_holdings.loc[metric_name, symbol]
            return np.nan if pd.isna(value) else float(value)
    except Exception:
        return np.nan
    return np.nan


def coerce_numeric_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a", "None"}:
        return np.nan

    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    elif text.endswith("T"):
        multiplier = 1_000_000_000_000
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1_000
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def extract_top_holding_weight(funds_data, symbol):
    try:
        top_holdings = funds_data.top_holdings
        if top_holdings is None or top_holdings.empty:
            return np.nan
    except Exception:
        return np.nan

    lookup_symbol = str(symbol).upper()
    normalized_columns = {str(col).strip().lower(): col for col in top_holdings.columns}
    symbol_col = normalized_columns.get("symbol") or normalized_columns.get("holding")
    weight_col = (
        normalized_columns.get("holdingpercent")
        or normalized_columns.get("holding percent")
        or normalized_columns.get("weight")
        or normalized_columns.get("portfoliopercent")
    )

    try:
        if symbol_col and weight_col:
            matches = top_holdings[top_holdings[symbol_col].astype(str).str.upper() == lookup_symbol]
            if not matches.empty:
                weight = coerce_numeric_value(matches.iloc[0][weight_col])
                if pd.notna(weight):
                    return weight * 100 if weight <= 1 else weight

        if lookup_symbol in top_holdings.index.astype(str).str.upper():
            row = top_holdings.loc[top_holdings.index.astype(str).str.upper() == lookup_symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for candidate in ["holdingPercent", "holding percent", "weight", "portfolioPercent"]:
                if candidate in row.index:
                    weight = coerce_numeric_value(row[candidate])
                    if pd.notna(weight):
                        return weight * 100 if weight <= 1 else weight
    except Exception:
        return np.nan
    return np.nan


def estimate_forward_pe(info, earnings_estimate, eps_trend, current_price, trailing_pe):
    forward_pe = info.get("forwardPE", np.nan)
    if pd.notna(forward_pe):
        return forward_pe, "info"

    forward_eps = info.get("forwardEps", np.nan)
    if pd.notna(current_price) and pd.notna(forward_eps) and forward_eps not in [0, np.nan]:
        return current_price / forward_eps, "derived_from_forward_eps"

    for table in [earnings_estimate, eps_trend]:
        if isinstance(table, pd.DataFrame) and not table.empty:
            cols = [col for col in table.columns if str(col).lower() in {"avg", "current"}]
            if cols:
                try:
                    eps_candidate = pd.to_numeric(table[cols[0]], errors="coerce").dropna()
                    if not eps_candidate.empty and pd.notna(current_price) and eps_candidate.iloc[0] != 0:
                        return current_price / eps_candidate.iloc[0], "derived_from_estimate_table"
                except Exception:
                    pass

    if pd.notna(trailing_pe):
        return trailing_pe, "proxy_from_trailing_pe"
    return np.nan, "unavailable"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_analyst_view(symbol):
    ticker = yf.Ticker(symbol)
    recommendations = safe_get_attr(ticker, "recommendations", pd.DataFrame())
    recommendations_summary = safe_get_attr(ticker, "recommendations_summary", pd.DataFrame())
    upgrades_downgrades = safe_get_attr(ticker, "upgrades_downgrades", pd.DataFrame())
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        recommendations = recommendations.tail(12)
    if isinstance(upgrades_downgrades, pd.DataFrame) and not upgrades_downgrades.empty:
        upgrades_downgrades = upgrades_downgrades.tail(12)
    return {
        "recommendations": recommendations if isinstance(recommendations, pd.DataFrame) else pd.DataFrame(),
        "recommendations_summary": recommendations_summary if isinstance(recommendations_summary, pd.DataFrame) else pd.DataFrame(),
        "upgrades_downgrades": upgrades_downgrades if isinstance(upgrades_downgrades, pd.DataFrame) else pd.DataFrame(),
    }


def prepare_recommendation_summary(summary_df):
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return pd.DataFrame(), {}

    df = summary_df.copy()
    if "period" in df.columns:
        df = df.set_index("period")
    df.index = [str(idx) for idx in df.index]

    score_map = {
        "strongBuy": 5,
        "buy": 4,
        "hold": 3,
        "sell": 2,
        "strongSell": 1,
    }
    available_cols = [col for col in ["strongBuy", "buy", "hold", "sell", "strongSell"] if col in df.columns]
    latest = df.iloc[0] if not df.empty else pd.Series(dtype=float)
    total = float(sum(pd.to_numeric(latest.get(col, 0), errors="coerce") for col in available_cols)) if not latest.empty else 0
    weighted_sum = sum(pd.to_numeric(latest.get(col, 0), errors="coerce") * score_map[col] for col in available_cols)
    avg_score = weighted_sum / total if total else np.nan

    sentiment = "Neutral"
    if pd.notna(avg_score):
        if avg_score >= 4.2:
            sentiment = "Strong Buy Bias"
        elif avg_score >= 3.5:
            sentiment = "Buy Bias"
        elif avg_score >= 2.5:
            sentiment = "Hold / Mixed"
        else:
            sentiment = "Sell Bias"

    return df, {
        "total_analysts": int(total) if total else 0,
        "avg_score": avg_score,
        "sentiment": sentiment,
        "buy_count": int(pd.to_numeric(latest.get("buy", 0), errors="coerce") + pd.to_numeric(latest.get("strongBuy", 0), errors="coerce")) if not latest.empty else 0,
        "hold_count": int(pd.to_numeric(latest.get("hold", 0), errors="coerce")) if not latest.empty else 0,
        "sell_count": int(pd.to_numeric(latest.get("sell", 0), errors="coerce") + pd.to_numeric(latest.get("strongSell", 0), errors="coerce")) if not latest.empty else 0,
    }


def prepare_upgrades_table(upgrades_df):
    if not isinstance(upgrades_df, pd.DataFrame) or upgrades_df.empty:
        return pd.DataFrame()
    df = upgrades_df.copy().reset_index(drop=False)
    rename_map = {
        "GradeDate": "Date",
        "date": "Date",
        "Firm": "Firm",
        "firm": "Firm",
        "ToGrade": "To Grade",
        "FromGrade": "From Grade",
        "Action": "Action",
    }
    df = df.rename(columns=rename_map)
    keep = [col for col in ["Date", "Firm", "To Grade", "From Grade", "Action"] if col in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep].tail(12).iloc[::-1].reset_index(drop=True)


def extract_estimate_point(table, keywords, preferred_cols=("Avg", "Current", "Low", "High")):
    if not isinstance(table, pd.DataFrame) or table.empty:
        return np.nan

    normalized_keywords = [keyword.lower() for keyword in keywords]
    for idx in table.index:
        idx_text = str(idx).lower().replace("_", " ").strip()
        if any(keyword in idx_text for keyword in normalized_keywords):
            row = table.loc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for col in preferred_cols:
                if col in row.index:
                    value = coerce_numeric_value(row[col])
                    if pd.notna(value):
                        return value
            for value in row.values:
                numeric = coerce_numeric_value(value)
                if pd.notna(numeric):
                    return numeric
    return np.nan


def extract_series_derivatives(series_values):
    clean = pd.Series(series_values, dtype="float64").dropna()
    if len(clean) < 3:
        return np.nan, np.nan
    x = np.arange(len(clean), dtype=float)
    slope = np.polyfit(x, clean.values, 1)[0]
    accel = np.polyfit(x, clean.values, 2)[0] * 2
    return slope, accel


def classify_range_location(current_price, low_52w, high_52w):
    if any(pd.isna(v) for v in [current_price, low_52w, high_52w]) or high_52w <= low_52w:
        return np.nan, "Δεν υπάρχει καθαρή τοποθέτηση μέσα στο 52-week range."
    position = ((current_price - low_52w) / (high_52w - low_52w)) * 100
    if position >= 80:
        text = "Η μετοχή κάθεται ψηλά στο 52-week range, άρα η αγορά ήδη τιμολογεί αρκετή αισιοδοξία."
    elif position >= 55:
        text = "Η μετοχή βρίσκεται στο upper half του 52-week range χωρίς να είναι ήδη σε ακραία υπερτιμημένη ζώνη."
    elif position >= 35:
        text = "Η μετοχή κινείται κοντά στη μέση του 52-week range και αφήνει ανοιχτό χώρο και προς τις δύο κατευθύνσεις."
    else:
        text = "Η μετοχή κάθεται χαμηλά στο 52-week range, κάτι που συνήθως σημαίνει ότι η αγορά ζητά πρώτα καθαρή επιβεβαίωση."
    return position, text


def build_benchmark_sensitivity_text(analysis):
    benchmark = analysis["benchmark"]
    if benchmark not in {"SPY", "QQQ"}:
        return ""

    membership = analysis.get("benchmark_membership", {})
    weight_pct = membership.get("weight_pct", np.nan)
    coint_pval = analysis["latest_metrics"].get("coint_pval", np.nan)
    if pd.isna(weight_pct) or weight_pct < 1.0:
        return ""
    if pd.isna(coint_pval) or coint_pval > 0.10:
        return ""

    financials = analysis["financials"]
    quarterly = financials["quarterly"]
    revenue_row = pd.to_numeric(quarterly.loc["Revenue"], errors="coerce") if "Revenue" in quarterly.index else pd.Series(dtype=float)
    revenue_history = revenue_row.iloc[::-1].tail(4) if not revenue_row.empty else pd.Series(dtype=float)
    revenue_growth = revenue_history.pct_change().replace([np.inf, -np.inf], np.nan) * 100 if len(revenue_history) >= 2 else pd.Series(dtype=float)
    revenue_slope, revenue_accel = extract_series_derivatives(revenue_growth)

    eps_row = pd.to_numeric(quarterly.loc["Diluted EPS"], errors="coerce") if "Diluted EPS" in quarterly.index else pd.Series(dtype=float)
    if eps_row.empty and "Net Income" in quarterly.index:
        eps_row = pd.to_numeric(quarterly.loc["Net Income"], errors="coerce")
    earnings_history = eps_row.iloc[::-1].tail(4) if not eps_row.empty else pd.Series(dtype=float)
    next_quarter_eps = extract_estimate_point(financials["earnings_estimate"], ["next qtr", "+1q", "next quarter", "1q"])
    if pd.notna(next_quarter_eps):
        earnings_history = pd.concat([earnings_history, pd.Series([next_quarter_eps])], ignore_index=True)
    earnings_growth = earnings_history.pct_change().replace([np.inf, -np.inf], np.nan) * 100 if len(earnings_history) >= 2 else pd.Series(dtype=float)
    earnings_slope, earnings_accel = extract_series_derivatives(earnings_growth)

    next_quarter_revenue = extract_estimate_point(financials["revenue_estimate"], ["next qtr", "+1q", "next quarter", "1q"])
    next_year_revenue = extract_estimate_point(financials["revenue_estimate"], ["next year", "+1y", "next year revenue", "1y"])

    live = analysis["live_briefing"]
    low_52w = live["daily"]["Low"].dropna().min() if "Low" in live["daily"] else np.nan
    high_52w = live["daily"]["High"].dropna().max() if "High" in live["daily"] else np.nan
    range_position, range_text = classify_range_location(live["current_price"], low_52w, high_52w)

    if pd.isna(earnings_slope):
        return ""

    if earnings_slope > 4 and earnings_accel > 0:
        trend_text = "Η καμπύλη του earnings growth ανεβαίνει και επιταχύνει, άρα το fundamental trend παραμένει υπέρ των buyers."
        handling_text = "Για έναν απλό αγοραστή αυτό σημαίνει ότι η θέση μπορεί να παραμείνει σε watchlist ή και σε διακράτηση, αρκεί η μετοχή να μη χάσει βίαια τη δομή της τους επόμενους μήνες."
    elif earnings_slope > 0 and earnings_accel <= 0:
        trend_text = "Το earnings growth παραμένει θετικό, αλλά η δεύτερη παράγωγος γυρίζει χαμηλότερα και δείχνει ότι η βελτίωση ωριμάζει."
        handling_text = "Σε αυτή την περίπτωση πιο συνετό είναι να κυνηγά κανείς επιβεβαίωση στα αποτελέσματα του επόμενου quarter αντί για επιθετικό chasing."
    elif earnings_slope <= 0 and earnings_accel < 0:
        trend_text = "Το earnings growth χάνει κλίση και επιδεινώνεται, άρα η αγορά μπορεί να γίνει πιο αυστηρή απέναντι στο valuation."
        handling_text = "Για έναν απλό επενδυτή αυτό ταιριάζει περισσότερο σε αμυντική στάση, μικρότερη έκθεση ή αναμονή μέχρι να σταθεροποιηθεί η επιχειρηματική τάση."
    else:
        trend_text = "Το earnings trend δεν καταρρέει, αλλά δεν δίνει και καθαρό acceleration signal."
        handling_text = "Η πιο ισορροπημένη διαχείριση εδώ είναι η υπομονή και η παρακολούθηση των επόμενων reports πριν αυξηθεί το ρίσκο."

    forward_text_parts = []
    if pd.notna(next_quarter_revenue):
        forward_text_parts.append(f"Το next-quarter revenue estimate κάθεται γύρω στα {format_human_value(next_quarter_revenue)}")
    if pd.notna(next_year_revenue):
        forward_text_parts.append(f"ενώ το next-year revenue estimate διαμορφώνεται κοντά στα {format_human_value(next_year_revenue)}")
    forward_text = ", ".join(forward_text_parts) + "." if forward_text_parts else "Δεν υπήρχε καθαρό forward revenue range από το feed για να ενσωματωθεί πλήρως."

    revenue_text = ""
    if pd.notna(revenue_slope):
        if revenue_slope > 0:
            revenue_text = "Η κλίση των revenue growth observations των τελευταίων τεσσάρων τριμήνων παραμένει θετική."
        else:
            revenue_text = "Η κλίση των revenue growth observations των τελευταίων τεσσάρων τριμήνων έχει αρχίσει να γυρίζει χαμηλότερα."

    range_position_text = (
        f"στο {range_position:.1f}% του 52-week range"
        if pd.notna(range_position)
        else "μέσα στο διαθέσιμο 52-week range"
    )

    return (
        f"Index-sensitive note: η {analysis['ticker']} είναι ουσιαστικό component του {benchmark} με βάρος περίπου {weight_pct:.2f}% και "
        f"η σχέση της με τον δείκτη περνά το φίλτρο cointegration με p-value {coint_pval:.3f}. {range_text} "
        f"{forward_text} {revenue_text} {trend_text} {handling_text} "
        f"Με απλά λόγια, επειδή η μετοχή επηρεάζει αισθητά τον ίδιο τον δείκτη, το επόμενο quarter και το επόμενο year guidance έχουν μεγαλύτερη σημασία από ό,τι σε ένα μικρό component, ειδικά όταν η τιμή βρίσκεται {range_position_text}."
    )


def format_display_table(dataframe):
    if dataframe.empty:
        return dataframe
    formatted = dataframe.copy().astype(object)
    for row_name in formatted.index:
        row_label = str(row_name).lower()
        if "margin" in row_label or "growth" in row_label:
            formatted.loc[row_name] = [("-" if pd.isna(v) else f"{float(v):.2f}%") for v in formatted.loc[row_name]]
        elif "analyst" in row_label:
            formatted.loc[row_name] = [("-" if pd.isna(v) else f"{int(round(float(v)))}") for v in formatted.loc[row_name]]
        else:
            formatted.loc[row_name] = [format_human_value(v) for v in formatted.loc[row_name]]
    return formatted


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty or "Close" not in raw:
        return pd.DataFrame()
    close = raw["Close"].dropna(how="all")
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_pulse_data(ticker, range_key):
    config = determine_period_config(range_key)
    data = yf.download(
        ticker,
        period=config["period"],
        interval=config["interval"],
        prepost=True,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise ValueError(f"Δεν επέστρεψαν pulse δεδομένα για {ticker} στο {range_key}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data.dropna(how="all"), config


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_briefing_data(ticker):
    daily = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    intraday = yf.download(ticker, period="5d", interval="30m", prepost=True, progress=False, auto_adjust=False)
    if daily.empty:
        raise ValueError(f"Δεν επέστρεψαν daily δεδομένα για {ticker}.")

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = [col[0] for col in daily.columns]
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = [col[0] for col in intraday.columns]

    daily = daily.dropna(how="all")
    intraday = intraday.dropna(how="all")
    close = daily["Close"].dropna()
    high = daily["High"].dropna()
    low = daily["Low"].dropna()
    volume = daily["Volume"].dropna() if "Volume" in daily else pd.Series(dtype=float)

    current_price = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else current_price
    ret_1d = (current_price / prev_close - 1) * 100 if prev_close else np.nan
    ret_1w = (current_price / close.iloc[max(0, len(close) - 6)] - 1) * 100 if len(close) > 5 else np.nan
    ret_1m = (current_price / close.iloc[max(0, len(close) - 22)] - 1) * 100 if len(close) > 21 else np.nan

    year_start = close[close.index >= pd.Timestamp(dt.date.today().replace(month=1, day=1))]
    ytd_ret = (current_price / year_start.iloc[0] - 1) * 100 if not year_start.empty else np.nan

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    realized_vol_20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    atr14 = tr_components.max(axis=1).rolling(14).mean().iloc[-1]

    latest_volume = volume.iloc[-1] if not volume.empty else np.nan
    volume_ratio = (
        latest_volume / volume.rolling(20).mean().iloc[-1]
        if len(volume) > 20 and volume.rolling(20).mean().iloc[-1]
        else np.nan
    )

    high_52w = high.iloc[-252:].max() if len(high) >= 1 else np.nan
    low_52w = low.iloc[-252:].min() if len(low) >= 1 else np.nan
    dist_high = (current_price / high_52w - 1) * 100 if pd.notna(high_52w) and high_52w else np.nan
    dist_low = (current_price / low_52w - 1) * 100 if pd.notna(low_52w) and low_52w else np.nan

    intraday_close = intraday["Close"].dropna() if not intraday.empty and "Close" in intraday else pd.Series(dtype=float)
    pulse_return = (intraday_close.iloc[-1] / intraday_close.iloc[0] - 1) * 100 if len(intraday_close) > 1 else np.nan

    trend_score = sum(
        [
            1 if current_price > sma20.iloc[-1] else 0,
            1 if current_price > sma50.iloc[-1] else 0,
            1 if pd.notna(sma200.iloc[-1]) and current_price > sma200.iloc[-1] else 0,
        ]
    )

    return {
        "ticker": ticker,
        "daily": daily,
        "intraday": intraday,
        "current_price": current_price,
        "ret_1d": ret_1d,
        "ret_1w": ret_1w,
        "ret_1m": ret_1m,
        "ytd_ret": ytd_ret,
        "realized_vol_20": realized_vol_20,
        "atr14": atr14,
        "volume_ratio": volume_ratio,
        "dist_high": dist_high,
        "dist_low": dist_low,
        "pulse_return": pulse_return,
        "trend_score": trend_score,
        "updated_at": dt.datetime.now(),
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_range_reference_data(ticker):
    daily = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    if daily.empty:
        return pd.DataFrame()
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = [col[0] for col in daily.columns]
    return daily.dropna(how="all")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark_membership_data(benchmark_symbol, component_symbol):
    ticker = yf.Ticker(benchmark_symbol)
    funds_data = safe_get_attr(ticker, "funds_data", None)
    weight_pct = extract_top_holding_weight(funds_data, component_symbol)
    return {
        "benchmark": benchmark_symbol,
        "component": component_symbol,
        "weight_pct": weight_pct,
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals_data(symbol):
    ticker = yf.Ticker(symbol)
    info = safe_get_attr(ticker, "info", {}) or {}
    funds_data = safe_get_attr(ticker, "funds_data", None)

    quarterly_income = safe_get_attr(ticker, "quarterly_income_stmt", pd.DataFrame())
    annual_income = safe_get_attr(ticker, "income_stmt", pd.DataFrame())
    quarterly_cashflow = safe_get_attr(ticker, "quarterly_cashflow", pd.DataFrame())
    annual_cashflow = safe_get_attr(ticker, "cashflow", pd.DataFrame())

    quarterly_rows = {
        "Revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
        "Gross Profit": ["Gross Profit"],
        "Operating Income": ["Operating Income"],
        "Net Income": ["Net Income", "Net Income Common Stockholders"],
        "Diluted EPS": ["Diluted EPS", "Basic EPS"],
        "CapEx": ["Capital Expenditure"],
    }
    annual_rows = dict(quarterly_rows)

    quarterly = build_statement_table(quarterly_income, quarterly_rows, periods=4)
    annual = build_statement_table(annual_income, annual_rows, periods=4)

    if "CapEx" not in quarterly.index and quarterly_cashflow is not None and not quarterly_cashflow.empty:
        quarterly = pd.concat([quarterly, build_statement_table(quarterly_cashflow, {"CapEx": ["Capital Expenditure"]}, periods=4)])
    if "CapEx" not in annual.index and annual_cashflow is not None and not annual_cashflow.empty:
        annual = pd.concat([annual, build_statement_table(annual_cashflow, {"CapEx": ["Capital Expenditure"]}, periods=4)])

    quarterly = add_margin_rows(quarterly)
    annual = add_margin_rows(annual)

    earnings_estimate = normalize_estimate_table(safe_get_attr(ticker, "earnings_estimate", pd.DataFrame()))
    revenue_estimate = normalize_estimate_table(safe_get_attr(ticker, "revenue_estimate", pd.DataFrame()))
    eps_trend = normalize_estimate_table(safe_get_attr(ticker, "eps_trend", pd.DataFrame()))

    latest_quarter_period = quarterly.columns[0] if not quarterly.empty else "-"
    latest_quarter = quarterly.iloc[:, 0] if not quarterly.empty else pd.Series(dtype=float)
    current_price = info.get("regularMarketPrice", np.nan)
    trailing_eps = info.get("trailingEps", np.nan)
    forward_eps = info.get("forwardEps", np.nan)
    trailing_pe = info.get("trailingPE", np.nan)
    ps_ratio = info.get("priceToSalesTrailing12Months", np.nan)

    if pd.isna(trailing_pe):
        trailing_pe = extract_fund_equity_ratio(funds_data, "Price/Earnings", symbol)
    if pd.isna(ps_ratio):
        ps_ratio = extract_fund_equity_ratio(funds_data, "Price/Sales", symbol)

    if pd.isna(trailing_pe) and pd.notna(current_price) and pd.notna(trailing_eps) and trailing_eps != 0:
        trailing_pe = current_price / trailing_eps

    forward_pe, forward_pe_source = estimate_forward_pe(info, earnings_estimate, eps_trend, current_price, trailing_pe)

    snapshot = {
        "current_price": current_price,
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "ps_ratio": ps_ratio,
        "revenue": pick_first_available(latest_quarter, ["Revenue"]),
        "net_income": pick_first_available(latest_quarter, ["Net Income"]),
        "capex": pick_first_available(latest_quarter, ["CapEx"]),
        "profit_margin": pick_first_available(latest_quarter, ["Profit Margin %"]),
        "net_profit_margin": pick_first_available(latest_quarter, ["Net Profit Margin %"]),
        "last_quarter": latest_quarter_period,
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forward_pe_source": forward_pe_source,
    }

    return {
        "info": info,
        "snapshot": snapshot,
        "quarterly": quarterly,
        "annual": annual,
        "earnings_estimate": earnings_estimate,
        "revenue_estimate": revenue_estimate,
        "eps_trend": eps_trend,
    }


def build_market_pulse_commentary(ticker, range_key, hist):
    close = hist["Close"].dropna()
    if close.empty or len(close) < 2:
        return f"Δεν υπάρχουν αρκετά δεδομένα για pulse ανάλυση στο {range_key} για τη {ticker}."

    start_price = close.iloc[0]
    current_price = close.iloc[-1]
    change_pct = (current_price / start_price - 1) * 100
    returns = close.pct_change().dropna()
    # Προσαρμόζουμε τον annualization factor ανάλογα με το interval
    # (5m bars → ~78/day, 30m → ~13/day, 1h → ~7/day, 1d → 1/day)
    time_span_days = (hist.index[-1] - hist.index[0]).days if (hist.index[-1] - hist.index[0]).days > 0 else 1
    observations_per_day = len(close) / time_span_days if time_span_days > 0 else 78
    annualization_factor = np.sqrt(252 * max(1, observations_per_day))
    realized_vol = returns.std() * annualization_factor * 100 if not returns.empty else 0

    high = hist["High"].dropna().max() if "High" in hist else current_price
    low = hist["Low"].dropna().min() if "Low" in hist else current_price
    session_range_pct = (high / low - 1) * 100 if low else 0

    if change_pct > 2:
        direction = f"Η {ticker} κινείται έντονα ανοδικά στο {range_key}, περίπου στο +{change_pct:.2f}%."
    elif change_pct > 0:
        direction = f"Η {ticker} παραμένει θετική στο {range_key}, περίπου στο +{change_pct:.2f}%."
    elif change_pct < -2:
        direction = f"Η {ticker} πιέζεται αρκετά στο {range_key}, περίπου στο {change_pct:.2f}%."
    else:
        direction = f"Η {ticker} είναι ελαφρώς αρνητική στο {range_key}, περίπου στο {change_pct:.2f}%."

    if realized_vol > 55:
        vol_text = f"Η volatility είναι πολύ υψηλή ({realized_vol:.1f}%), άρα το tape είναι νευρικό και πιο επιθετικό."
    elif realized_vol > 30:
        vol_text = f"Η volatility είναι elevated ({realized_vol:.1f}%), οπότε το setup θέλει tighter risk management."
    else:
        vol_text = f"Η volatility είναι πιο orderly ({realized_vol:.1f}%), κάτι που βοηθά καθαρότερη ανάγνωση τάσης."

    structure = (
        "Η τιμή κρατά πάνω από τη μέση τροχιά του επιλεγμένου window, οπότε οι buyers διατηρούν σχετικό έλεγχο."
        if current_price >= close.mean()
        else "Η τιμή κινείται κάτω από τη μέση τροχιά του επιλεγμένου window, άρα το selling pressure είναι πιο ορατό."
    )

    p1 = f"{direction} {vol_text} Το range του επιλεγμένου παραθύρου είναι {session_range_pct:.2f}% με τιμές από {low:.2f} έως {high:.2f}."
    p2 = (
        f"{structure} Αν συνεχίσει να κρατά higher lows, η βραχυπρόθεσμη τάση μένει υπέρ των buyers. "
        "Αν όμως η μεταβλητότητα ανοίγει μαζί με πτώση της τιμής, ενισχύεται το selling pressure, ενώ "
        "σταθεροποίηση της τιμής με χαμηλότερη volatility συνήθως βοηθά να χτιστεί πιο υγιής βάση."
    )
    return "\n\n".join([p1, p2])


def build_live_briefing_text(data):
    ticker = data["ticker"]
    if data["trend_score"] >= 3:
        trend_text = f"Η {ticker} κρατά SMA 20, 50 και 200, οπότε το trend profile είναι ξεκάθαρα constructive."
    elif data["trend_score"] == 2:
        trend_text = f"Η {ticker} παραμένει constructive, αλλά όχι πλήρως dominant σε όλα τα trend layers."
    else:
        trend_text = f"Η {ticker} χάνει αρκετά trend layers και το momentum χρειάζεται προσοχή."

    vol = data["realized_vol_20"]
    if pd.notna(vol) and vol > 50:
        vol_text = f"Η 20-day realized volatility είναι {vol:.1f}%, άρα η μετοχή κινείται σε επιθετικό risk bucket."
    elif pd.notna(vol) and vol > 28:
        vol_text = f"Η 20-day realized volatility είναι {vol:.1f}%, δηλαδή elevated αλλά ακόμα διαχειρίσιμη."
    else:
        vol_text = f"Η 20-day realized volatility είναι {vol:.1f}%, κάτι που δείχνει πιο orderly tape."

    volume_text = (
        f"Ο όγκος είναι {data['volume_ratio']:.2f}x του 20-day average." if pd.notna(data["volume_ratio"]) else "Δεν υπάρχει καθαρό volume ratio."
    )
    location_text = (
        f"Η τιμή απέχει {abs(data['dist_high']):.2f}% από το 52-week high και βρίσκεται {data['dist_low']:.2f}% πάνω από το 52-week low."
        if pd.notna(data["dist_high"]) and pd.notna(data["dist_low"])
        else "Δεν κατέστη δυνατό να χαρτογραφηθεί πλήρως το 52-week range."
    )
    p1 = f"Live briefing για {ticker}: τελευταία τιμή {data['current_price']:.2f}. {trend_text} {vol_text} {volume_text} {location_text}"
    p2 = (
        f"Η βραχυπρόθεσμη εικόνα γράφει 1D {data['ret_1d']:+.2f}%, 1W {data['ret_1w']:+.2f}%, 1M {data['ret_1m']:+.2f}% και "
        f"YTD {data['ytd_ret']:+.2f}%, ενώ το intraday pulse είναι {data['pulse_return']:+.2f}% και το ATR(14) {data['atr14']:.2f}. "
        "Όταν η τιμή κρατά πάνω από SMA 20 με ισχυρότερο όγκο, οι αγοραστές παραμένουν σε πλεονέκτημα. Αν όμως η realized volatility "
        "ανεβαίνει ενώ το weekly momentum γυρίζει αρνητικό, το setup μπαίνει σε πιο δύσκολη ζώνη."
    )
    return "\n\n".join([p1, p2])


def build_regime_text(ticker, benchmark, benchmark_bull, stock_bull):
    if benchmark_bull and stock_bull:
        return "Bull market alignment: αγορά και μετοχή κινούνται πάνω από τον 200-day SMA, κάτι που ευνοεί continuation και trend following."
    if benchmark_bull and not stock_bull:
        return f"Relative weakness: ο {benchmark} κρατά ανοδική τάση, αλλά η {ticker} όχι, κάτι που συχνά δείχνει rotation ή εσωτερική αδυναμία."
    if not benchmark_bull and stock_bull:
        return f"Relative strength: η αγορά είναι πιο αδύναμη, αλλά η {ticker} κρατιέται πάνω από τον 200-day SMA και συμπεριφέρεται σαν ηγέτης."
    return "Bear market pressure: τόσο η αγορά όσο και η μετοχή είναι κάτω από τον 200-day SMA, οπότε το setup θέλει πιο αμυντικό risk control."


def build_volatility_text(vol_ratio):
    if pd.isna(vol_ratio):
        return "Δεν υπάρχει καθαρό volatility ratio ακόμα."
    if vol_ratio > 1.35:
        return "Η βραχυπρόθεσμη μεταβλητότητα έχει εκτοξευθεί πάνω από τη μακροπρόθεσμη και το tape είναι σαφώς πιο νευρικό."
    if vol_ratio > 1.1:
        return "Η μεταβλητότητα έχει αρχίσει να ανεβαίνει και το asset μπαίνει σε πιο απαιτητικό risk regime."
    return "Η βραχυπρόθεσμη μεταβλητότητα παραμένει ελεγχόμενη σε σχέση με τη μεγάλη εικόνα."


def build_valuation_summary(stock_symbol, benchmark_symbol, stock_fundamentals, benchmark_fundamentals, spy_fundamentals):
    stock_snapshot = stock_fundamentals["snapshot"]
    benchmark_snapshot = benchmark_fundamentals["snapshot"]
    spy_snapshot = spy_fundamentals["snapshot"]

    return [
        {
            "title": f"{stock_symbol} Trailing P/E",
            "value": format_human_value(stock_snapshot["trailing_pe"]),
            "subtitle": f"{benchmark_symbol}: {format_human_value(benchmark_snapshot['trailing_pe'])} | SPY: {format_human_value(spy_snapshot['trailing_pe'])}",
            "delta": f"vs {benchmark_symbol}: {metric_delta_text(stock_snapshot['trailing_pe'], benchmark_snapshot['trailing_pe'])}",
            "symbol": stock_symbol,
            "metric_key": "trailing_pe",
        },
        {
            "title": f"{stock_symbol} Forward P/E",
            "value": format_human_value(stock_snapshot["forward_pe"]),
            "subtitle": f"{benchmark_symbol}: {format_human_value(benchmark_snapshot['forward_pe'])} | SPY: {format_human_value(spy_snapshot['forward_pe'])}",
            "delta": f"vs {benchmark_symbol}: {metric_delta_text(stock_snapshot['forward_pe'], benchmark_snapshot['forward_pe'])}",
            "symbol": stock_symbol,
            "metric_key": "forward_pe",
        },
        {
            "title": "Forward EPS",
            "value": format_human_value(stock_snapshot["forward_eps"]),
            "subtitle": f"{benchmark_symbol}: {format_human_value(benchmark_snapshot['forward_eps'])} | SPY: {format_human_value(spy_snapshot['forward_eps'])}",
            "delta": f"Trailing EPS: {format_human_value(stock_snapshot['trailing_eps'])}",
            "symbol": stock_symbol,
            "metric_key": "forward_eps",
        },
        {
            "title": "P/S Ratio",
            "value": format_human_value(stock_snapshot["ps_ratio"]),
            "subtitle": f"{benchmark_symbol}: {format_human_value(benchmark_snapshot['ps_ratio'])} | SPY: {format_human_value(spy_snapshot['ps_ratio'])}",
            "delta": f"Latest quarter: {stock_snapshot['last_quarter']}",
            "symbol": stock_symbol,
            "metric_key": "ps_ratio",
        },
        {
            "title": "Quarter Profit Margin",
            "value": format_percent(stock_snapshot["profit_margin"]),
            "subtitle": f"Net margin: {format_percent(stock_snapshot['net_profit_margin'])}",
            "delta": f"Revenue: {format_human_value(stock_snapshot['revenue'])}",
            "symbol": stock_symbol,
            "metric_key": "profit_margin",
        },
        {
            "title": "CapEx / Net Income",
            "value": format_human_value(stock_snapshot["capex"]),
            "subtitle": f"Net income: {format_human_value(stock_snapshot['net_income'])}",
            "delta": f"Updated: {stock_snapshot['updated']}",
            "symbol": stock_symbol,
            "metric_key": "capex",
        },
    ]


def build_analysis(ticker, benchmark="QQQ", beta_window=60):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * 2)
    tickers = list(dict.fromkeys([ticker, benchmark] + MAG7))
    close_prices = download_prices(tickers, start_date, end_date)

    if close_prices.empty:
        raise ValueError("Δεν επέστρεψαν price δεδομένα από το Yahoo Finance.")
    if ticker not in close_prices.columns or benchmark not in close_prices.columns:
        raise ValueError("Το ticker ή το benchmark δεν βρέθηκε στα downloaded δεδομένα.")

    stock = close_prices[ticker].dropna()
    benchmark_series = close_prices[benchmark].dropna()
    mag7_prices = close_prices[[symbol for symbol in MAG7 if symbol in close_prices.columns]].dropna()
    if len(stock) < 260 or len(benchmark_series) < 260 or mag7_prices.empty:
        raise ValueError("Χρειάζονται περισσότερα ιστορικά δεδομένα για σταθερή ανάλυση.")

    ret_stock = stock.pct_change().dropna()
    ret_benchmark = benchmark_series.pct_change().dropna()
    mag7_ret = mag7_prices.pct_change().dropna().mean(axis=1)

    aligned = pd.concat([ret_stock, ret_benchmark, mag7_ret], axis=1, join="inner")
    aligned.columns = ["ret_stock", "ret_benchmark", "mag7_ret"]
    ret_stock = aligned["ret_stock"]
    ret_benchmark = aligned["ret_benchmark"]
    mag7_ret = aligned["mag7_ret"]

    stock, benchmark_series = stock.align(benchmark_series, join="inner")
    normalized_stock = stock / stock.iloc[0] * 100
    normalized_benchmark = benchmark_series / benchmark_series.iloc[0] * 100
    relative_strength = normalized_stock / normalized_benchmark * 100

    sma50 = stock.rolling(50).mean()
    sma200 = stock.rolling(200).mean()
    benchmark_sma200 = benchmark_series.rolling(200).mean()
    regime_text = build_regime_text(
        ticker,
        benchmark,
        benchmark_series.iloc[-1] > benchmark_sma200.iloc[-1],
        stock.iloc[-1] > sma200.iloc[-1],
    )

    short_vol = ret_stock.rolling(30).std() * np.sqrt(252)
    long_vol = ret_stock.rolling(252).std() * np.sqrt(252)
    vol_ratio = safe_last(short_vol) / safe_last(long_vol) if pd.notna(safe_last(long_vol)) and safe_last(long_vol) != 0 else np.nan

    beta_benchmark = ret_stock.rolling(beta_window).cov(ret_benchmark) / ret_benchmark.rolling(beta_window).var()
    drawdown = (stock - stock.cummax()) / stock.cummax() * 100
    roll_sharpe = (ret_stock.rolling(60).mean() * 252 - 0.04) / (ret_stock.rolling(60).std() * np.sqrt(252))

    log_stock, log_benchmark = np.log(stock).align(np.log(benchmark_series), join="inner")
    slope, intercept = np.polyfit(log_benchmark.values, log_stock.values, 1)
    spread_series = log_stock - (intercept + slope * log_benchmark)
    spread_zscore = (spread_series - spread_series.mean()) / spread_series.std()

    try:
        coint_pval = coint(log_stock.values, log_benchmark.values)[1]
    except Exception:
        coint_pval = np.nan

    rsi = calculate_rsi(stock)
    current_price = stock.iloc[-1]

    stock_fundamentals = fetch_fundamentals_data(ticker)
    benchmark_fundamentals = fetch_fundamentals_data(benchmark)
    spy_fundamentals = fetch_fundamentals_data("SPY")
    benchmark_membership = fetch_benchmark_membership_data(benchmark, ticker) if benchmark in {"SPY", "QQQ"} else {"benchmark": benchmark, "component": ticker, "weight_pct": np.nan}

    latest_metrics = {
        "current_price": current_price,
        "latest_rsi": safe_last(rsi),
        "latest_sharpe": safe_last(roll_sharpe),
        "latest_beta_benchmark": safe_last(beta_benchmark),
        "latest_drawdown": safe_last(drawdown),
        "latest_zscore": safe_last(spread_zscore),
        "latest_short_vol": safe_last(short_vol) * 100,
        "latest_long_vol": safe_last(long_vol) * 100,
        "vol_ratio": vol_ratio,
        "coint_pval": coint_pval,
        "sma50": safe_last(sma50),
        "sma200": safe_last(sma200),
        "start_date": stock.index[0],
        "end_date": stock.index[-1],
    }

    pulse_data, pulse_config = fetch_market_pulse_data(ticker, "1D")
    live_briefing = fetch_live_briefing_data(ticker)

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "beta_window": beta_window,
        "stock": stock,
        "benchmark_series": benchmark_series,
        "normalized_stock": normalized_stock,
        "normalized_benchmark": normalized_benchmark,
        "relative_strength": relative_strength,
        "sma50": sma50,
        "sma200": sma200,
        "short_vol": short_vol,
        "long_vol": long_vol,
        "beta_benchmark": beta_benchmark,
        "drawdown": drawdown,
        "roll_sharpe": roll_sharpe,
        "spread_zscore": spread_zscore,
        "rsi": rsi,
        "regime_text": regime_text,
        "vol_text": build_volatility_text(vol_ratio),
        "latest_metrics": latest_metrics,
        "financials": stock_fundamentals,
        "benchmark_financials": benchmark_fundamentals,
        "spy_financials": spy_fundamentals,
        "benchmark_membership": benchmark_membership,
        "valuation_cards": build_valuation_summary(ticker, benchmark, stock_fundamentals, benchmark_fundamentals, spy_fundamentals),
        "pulse_data": pulse_data,
        "pulse_config": pulse_config,
        "live_briefing": live_briefing,
    }


def build_line_area_chart(df, x_col, y_col, color="#72f1ff", fill="#15b57a", title=None, minimal=False, x_type="temporal"):
    axis_x = None if minimal else alt.Axis(labelColor="#8EA0B8", title=None, grid=False)
    axis_y = None if minimal else alt.Axis(labelColor="#8EA0B8", title=None, grid=False)
    x_encoding = alt.X(f"{x_col}:{'T' if x_type == 'temporal' else 'Q'}", axis=axis_x)
    y_encoding = alt.Y(f"{y_col}:Q", axis=axis_y)

    base = alt.Chart(df).encode(x=x_encoding, y=y_encoding)
    area = base.mark_area(color=fill, opacity=0.16)
    line = base.mark_line(color=color, strokeWidth=2.8, interpolate="monotone")
    chart = (area + line).properties(height=320 if minimal else 360)
    if title:
        chart = chart.properties(title=title)
    return chart


def build_dual_line_chart(df, title):
    return (
        alt.Chart(df)
        .transform_fold(["Stock", "Benchmark"], as_=["Series", "Value"])
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(labelColor="#8EA0B8", title=None)),
            y=alt.Y("Value:Q", axis=alt.Axis(labelColor="#8EA0B8", title=None)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Stock", "Benchmark"], range=["#F59E0B", "#72F1FF"]),
                legend=alt.Legend(labelColor="#D7E2F0", title=None),
            ),
            tooltip=["Date:T", "Series:N", "Value:Q"],
        )
        .properties(height=320, title=title)
    )


def inject_styles():
    st.markdown(
        """
        <style>
        /* ── Deep Slate Terminal Theme ── */
        :root {
            --bg:       #131722;
            --panel:    #1c2030;
            --panel-2:  #1e2538;
            --border:   #2a2e39;
            --text:     #d1d4dc;
            --muted:    #787b86;
            --cyan:     #2962ff;
            --amber:    #f59e0b;
            --green:    #26a69a;
            --red:      #ef5350;
        }

        /* ── App shell ── */
        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: "Inter", "Helvetica Neue", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: #0f1219;
            border-right: 1px solid var(--border);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1520px;
        }

        /* ── Cards — flat, no gradients, no shadows ── */
        div[data-testid="stMetric"], .kpi-card, .glass-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            box-shadow: none;
        }
        .glass-card {
            padding: 0.75rem 0.9rem 0.7rem 0.9rem;
            margin-bottom: 0.65rem;
        }

        /* ── Hero banner ── */
        .hero {
            padding: 0.9rem 1.2rem 0.8rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--panel);
            margin-bottom: 0.9rem;
        }
        .hero h1 {
            margin: 0;
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        .hero p { margin: 0.3rem 0 0 0; color: var(--muted); line-height: 1.5; font-size: 0.88rem; }

        /* ── Section labels ── */
        .section-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.68rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.25rem;
        }

        /* ── KPI typography ── */
        .kpi-title {
            color: var(--muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 500;
        }
        .kpi-value {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.2rem;
            font-variant-numeric: tabular-nums;
        }
        .kpi-sub   { color: var(--muted); font-size: 0.78rem; margin-top: 0.3rem; line-height: 1.4; }
        .kpi-delta { color: var(--cyan);  font-size: 0.78rem; margin-top: 0.2rem; font-weight: 600; }

        /* ── Commentary / narrative cards ── */
        .commentary-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.85rem 1rem;
            color: var(--text);
            line-height: 1.7;
            white-space: pre-wrap;
            font-size: 0.875rem;
        }
        .small-note { color: var(--muted); font-size: 0.82rem; line-height: 1.55; }

        /* ── Dataframe cells ── */
        .dataframe tbody tr th,
        .dataframe tbody tr td { color: #d1d4dc !important; font-size: 0.82rem !important; }

        /* ── Tabs — pill → flat underline ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-radius: 0;
            border-bottom: 2px solid transparent;
            color: var(--muted);
            padding: 0.45rem 0.9rem;
            font-size: 0.82rem;
            font-weight: 500;
            margin-bottom: -1px;
        }
        .stTabs [aria-selected="true"] {
            background: transparent;
            color: var(--text);
            border-bottom: 2px solid var(--cyan);
        }

        /* ── Inputs ── */
        label,
        .stMarkdown,
        .stTextInput label,
        .stSelectbox label,
        .stNumberInput label { color: var(--muted) !important; font-size: 0.78rem !important; }

        div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            background: #0f1219 !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            font-size: 0.84rem !important;
        }

        /* ── Buttons ── */
        .stButton button {
            border-radius: 4px;
            background: var(--panel-2);
            color: var(--text);
            border: 1px solid var(--border);
            font-size: 0.82rem;
            font-weight: 500;
            padding: 0.35rem 0.8rem;
            transition: border-color 0.15s;
        }
        .stButton button:hover { border-color: var(--cyan); }

        /* ── P/L colour helpers (used in HTML injections) ── */
        .pl-pos { color: var(--green) !important; font-weight: 600; }
        .pl-neg { color: var(--red)   !important; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(ticker, benchmark):
    st.markdown(
        f"""
        <div class="hero">
            <div class="section-label">Enhanced Quant Terminal</div>
            <h1>{ticker} benchmarked with {benchmark}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_rail(cards):
    st.markdown('<div class="section-label">Valuation Rail</div>', unsafe_allow_html=True)
    for idx, card in enumerate(cards):
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{card['title']}</div>
                <div class="kpi-value">{card['value']}</div>
                <div class="kpi-sub">{card['subtitle']}</div>
                <div class="kpi-delta">{card['delta']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if card["metric_key"] in {"trailing_pe", "forward_pe"}:
            if st.button(f"Analyst view: {card['title']}", key=f"analyst_{idx}_{card['metric_key']}", width="stretch"):
                st.session_state.selected_analyst_symbol = card["symbol"]
                st.session_state.selected_analyst_metric = card["metric_key"]


def render_benchmark_multiples(benchmark_symbol, benchmark_fundamentals):
    snapshot = benchmark_fundamentals["snapshot"]
    st.markdown('<div class="section-label">Benchmark Multiples</div>', unsafe_allow_html=True)
    items = [
        ("Trailing P/E", format_human_value(snapshot["trailing_pe"])),
        ("Forward P/E", format_human_value(snapshot["forward_pe"])),
        ("P/S Ratio", format_human_value(snapshot["ps_ratio"])),
    ]
    for idx, (label, value) in enumerate(items):
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{benchmark_symbol} {label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">Updated: {snapshot['updated']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if label in {"Trailing P/E", "Forward P/E"}:
            metric_key = "trailing_pe" if "Trailing" in label else "forward_pe"
            if st.button(f"Analyst view: {benchmark_symbol} {label}", key=f"bench_analyst_{idx}_{metric_key}", width="stretch"):
                st.session_state.selected_analyst_symbol = benchmark_symbol
                st.session_state.selected_analyst_metric = metric_key


def render_analyst_panel():
    symbol = st.session_state.get("selected_analyst_symbol")
    metric = st.session_state.get("selected_analyst_metric")
    if not symbol or not metric:
        return

    st.markdown('<div class="section-label">Analyst View</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="small-note">Live public analyst feed for {symbol} on {metric.replace("_", " ").upper()}. Για major firms όπως Morgan Stanley / JPMorgan, τα ονόματα εμφανίζονται μόνο αν τα δίνει το διαθέσιμο Yahoo feed. Δεν είναι πλήρες institutional terminal feed.</div>',
        unsafe_allow_html=True,
    )
    try:
        analyst_data = fetch_analyst_view(symbol)
    except Exception as exc:
        st.markdown(f'<div class="commentary-card">Αδυναμία φόρτωσης analyst feed για {symbol}.\n\n{exc}</div>', unsafe_allow_html=True)
        return

    summary_table, summary_stats = prepare_recommendation_summary(analyst_data["recommendations_summary"])
    upgrades_table = prepare_upgrades_table(analyst_data["upgrades_downgrades"])

    if summary_stats:
        stat_cols = st.columns(4)
        stat_items = [
            ("Analysts", str(summary_stats["total_analysts"])),
            ("Buy", str(summary_stats["buy_count"])),
            ("Hold", str(summary_stats["hold_count"])),
            ("Sell", str(summary_stats["sell_count"])),
        ]
        for col, (label, value) in zip(stat_cols, stat_items):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="kpi-title">{label}</div>
                        <div class="kpi-value" style="font-size:1.05rem;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"""
            <div class="commentary-card">
                Consensus sentiment: {summary_stats['sentiment']}.
                Average weighted score: {"-" if pd.isna(summary_stats['avg_score']) else f"{summary_stats['avg_score']:.2f} / 5.00"}.
                Το summary αυτό βασίζεται στο public recommendation feed του Yahoo και λειτουργεί σαν terminal-style consensus snapshot.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(summary_table, width="stretch", height=170)

    if not upgrades_table.empty:
        st.markdown('<div class="section-label">Firm Actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Recent upgrades, downgrades ή maintained ratings από firms που εμφανίζονται στο διαθέσιμο analyst feed.</div>', unsafe_allow_html=True)
        st.dataframe(upgrades_table, width="stretch", height=260)
    elif not analyst_data["recommendations"].empty:
        st.markdown('<div class="section-label">Recommendation Tape</div>', unsafe_allow_html=True)
        st.dataframe(analyst_data["recommendations"], width="stretch", height=240)

    if not summary_stats and upgrades_table.empty and analyst_data["recommendations"].empty:
        st.markdown('<div class="commentary-card">Δεν υπάρχουν διαθέσιμα analyst rows από το δημόσιο feed για αυτό το symbol.</div>', unsafe_allow_html=True)


def render_range_bar(label, low_value, high_value, current_value):
    if any(pd.isna(v) for v in [low_value, high_value, current_value]) or high_value <= low_value:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="kpi-title">{label}</div>
                <div class="small-note">Range data not available.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    position = max(0.0, min(100.0, ((current_value - low_value) / (high_value - low_value)) * 100))
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-title">{label}</div>
            <div style="display:flex; justify-content:space-between; color:#8ea0b8; font-size:0.86rem; margin:0.25rem 0 0.55rem 0;">
                <span>{low_value:.2f}</span>
                <span>{high_value:.2f}</span>
            </div>
            <div style="position:relative; height:10px; border-radius:999px; background:#18222e; overflow:visible;">
                <div style="position:absolute; inset:0; border-radius:999px; background:linear-gradient(90deg, #1f2a36 0%, #253445 100%);"></div>
                <div style="position:absolute; left:calc({position}% - 7px); top:-4px; width:14px; height:14px; border-radius:999px; background:#ffffff; box-shadow:0 0 0 3px rgba(255,255,255,0.08);"></div>
            </div>
            <div style="margin-top:0.55rem; color:#dfe7f2; font-size:0.9rem;">Last close: {current_value:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(analysis):
    overview_df = pd.DataFrame(
        {
            "Date": analysis["stock"].index,
            "Price": analysis["stock"].values,
            "SMA 50": analysis["sma50"].values,
            "SMA 200": analysis["sma200"].values,
        }
    )
    rel_df = pd.DataFrame(
        {
            "Date": analysis["normalized_stock"].index,
            "Stock": analysis["normalized_stock"].values,
            "Benchmark": analysis["normalized_benchmark"].values,
            "Relative Strength": analysis["relative_strength"].values,
        }
    )

    top_left, top_right = st.columns([1.5, 1])
    with top_left:
        st.markdown('<div class="section-label">Trend Structure</div>', unsafe_allow_html=True)
        chart = (
            alt.Chart(overview_df)
            .transform_fold(["Price", "SMA 50", "SMA 200"], as_=["Series", "Value"])
            .mark_line(strokeWidth=2.4)
            .encode(
                x=alt.X("Date:T", axis=alt.Axis(labelColor="#8EA0B8", title=None)),
                y=alt.Y("Value:Q", axis=alt.Axis(labelColor="#8EA0B8", title=None)),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=["Price", "SMA 50", "SMA 200"], range=["#FFFFFF", "#15B57A", "#F59E0B"]),
                    legend=alt.Legend(labelColor="#D7E2F0", title=None),
                ),
                tooltip=["Date:T", "Series:N", "Value:Q"],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, width="stretch")
    with top_right:
        st.markdown('<div class="section-label">Narrative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="commentary-card">{wrap_text(analysis["regime_text"])}\n\n{wrap_text(analysis["vol_text"])}</div>', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown('<div class="section-label">Relative Performance</div>', unsafe_allow_html=True)
        st.altair_chart(build_dual_line_chart(rel_df[["Date", "Stock", "Benchmark"]], "Normalized Performance"), width="stretch")
    with bottom_right:
        st.markdown('<div class="section-label">Relative Strength</div>', unsafe_allow_html=True)
        rs_chart = build_line_area_chart(rel_df, "Date", "Relative Strength", color="#F59E0B", fill="#F59E0B", title=None, minimal=False)
        st.altair_chart(rs_chart, width="stretch")


def render_market_pulse_tab(ticker):
    range_key = st.segmented_control("Pulse Range", ["1D", "1W", "1M", "1Y"], default="1D", key="pulse_range")
    hist, config = fetch_market_pulse_data(ticker, range_key)
    reference_daily = fetch_range_reference_data(ticker)
    commentary = build_market_pulse_commentary(ticker, range_key, hist)
    close = hist["Close"].dropna()
    pulse_df = pd.DataFrame(
        {
            "Date": close.index,
            "Close": close.values,
        }
    )
    price_min = float(close.min())
    price_max = float(close.max())
    padding = max((price_max - price_min) * 0.18, max(price_max * 0.003, 0.25))
    domain_min = price_min - padding
    domain_max = price_max + padding
    latest_price = float(close.iloc[-1])
    latest_time = close.index[-1]
    latest_point_df = pd.DataFrame({"Date": [latest_time], "Close": [latest_price]})
    price_rule_df = pd.DataFrame({"Price": [latest_price]})
    latest_session = reference_daily.dropna(how="all").iloc[-1] if not reference_daily.empty else pd.Series(dtype=float)
    day_low = latest_session.get("Low", np.nan)
    day_high = latest_session.get("High", np.nan)
    close_52w = reference_daily["Close"].dropna() if not reference_daily.empty and "Close" in reference_daily else pd.Series(dtype=float)
    low_52w = close_52w.min() if not close_52w.empty else np.nan
    high_52w = close_52w.max() if not close_52w.empty else np.nan

    left, right = st.columns([2.1, 1])
    with left:
        st.markdown('<div class="section-label">Pulse Chart</div>', unsafe_allow_html=True)
        if range_key == "1D":
            x_axis = alt.Axis(format="%H:%M", labelColor="#8EA0B8", title=None, grid=False, tickCount=8)
        elif range_key == "1W":
            x_axis = alt.Axis(format="%d %b", labelColor="#8EA0B8", title=None, grid=False, tickCount=6)
        elif range_key == "1M":
            x_axis = alt.Axis(format="%d %b", labelColor="#8EA0B8", title=None, grid=False, tickCount=6)
        else:
            x_axis = alt.Axis(format="%b %y", labelColor="#8EA0B8", title=None, grid=False, tickCount=6)

        base = alt.Chart(pulse_df).encode(
            x=alt.X("Date:T", axis=x_axis),
            y=alt.Y(
                "Close:Q",
                axis=alt.Axis(labelColor="#8EA0B8", title=None, grid=False, tickCount=6),
                scale=alt.Scale(domain=[domain_min, domain_max], nice=False, zero=False),
            ),
            tooltip=[alt.Tooltip("Date:T", title="Time"), alt.Tooltip("Close:Q", format=".2f", title="Price")],
        )
        pulse_chart = (
            base.mark_area(color="#15B57A", opacity=0.12, clip=True)
            + base.mark_line(color="#FFFFFF", strokeWidth=3.0, interpolate="monotone")
            + alt.Chart(price_rule_df).mark_rule(color="#3A4757", strokeDash=[4, 4]).encode(y="Price:Q")
            + alt.Chart(latest_point_df).mark_point(color="#FFFFFF", filled=True, size=80).encode(x="Date:T", y="Close:Q")
        ).properties(height=340)
        st.altair_chart(pulse_chart, width="stretch")
        stats_cols = st.columns(4)
        change_pct = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else np.nan
        with stats_cols[0]:
            st.metric("Low", f"{price_min:.2f}")
        with stats_cols[1]:
            st.metric("High", f"{price_max:.2f}")
        with stats_cols[2]:
            st.metric("Last Price", f"{latest_price:.2f}")
        with stats_cols[3]:
            st.metric("Move", format_percent(change_pct))
        range_cols = st.columns(2)
        with range_cols[0]:
            render_range_bar("Day's Range", day_low, day_high, latest_price)
        with range_cols[1]:
            render_range_bar("52 Week Range", low_52w, high_52w, latest_price)
    with right:
        st.markdown('<div class="section-label">Coverage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="commentary-card">{wrap_text(commentary)}</div>', unsafe_allow_html=True)


def render_financials_tab(analysis):
    financials = analysis["financials"]
    snapshot = financials["snapshot"]

    st.markdown('<div class="section-label">Financial Snapshot</div>', unsafe_allow_html=True)
    cards = st.columns(8)
    card_items = [
        ("Last Quarter", snapshot["last_quarter"]),
        ("Trailing EPS", format_human_value(snapshot["trailing_eps"])),
        ("Forward EPS", format_human_value(snapshot["forward_eps"])),
        ("Trailing P/E", format_human_value(snapshot["trailing_pe"])),
        ("Forward P/E", format_human_value(snapshot["forward_pe"])),
        ("P/S", format_human_value(snapshot["ps_ratio"])),
        ("Profit Margin", format_percent(snapshot["profit_margin"])),
        ("Net Margin", format_percent(snapshot["net_profit_margin"])),
    ]
    for column, (label, value) in zip(cards, card_items):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{label}</div>
                    <div class="kpi-value" style="font-size:1.15rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="small-note">Η διάταξη παρακάτω χωρίζει καθαρά τα historical financials από τα forward estimates, ώστε να διαβάζονται πιο γρήγορα σαν research workspace.</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Quarterly", "Annual", "Forward Earnings", "Forward Revenue", "EPS Revisions"])

    with tabs[0]:
        st.markdown('<div class="section-label">Quarterly Financials</div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["quarterly"]), width="stretch", height=360)

    with tabs[1]:
        st.markdown('<div class="section-label">Annual Financials</div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["annual"]), width="stretch", height=360)

    with tabs[2]:
        st.markdown('<div class="section-label">Forward Earnings Estimates</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Range estimates για EPS / earnings με average, low, high και growth όταν τα δίνει το Yahoo.</div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["earnings_estimate"]), width="stretch", height=320)

    with tabs[3]:
        st.markdown('<div class="section-label">Forward Revenue Estimates</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Range estimates για revenue και expected growth σε διάταξη τύπου spreadsheet.</div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["revenue_estimate"]), width="stretch", height=320)

    with tabs[4]:
        st.markdown('<div class="section-label">EPS Trend Revisions</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Revision ladder για το forward EPS, ώστε να βλέπεις αν οι εκτιμήσεις ανεβαίνουν ή χαμηλώνουν.</div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["eps_trend"]), width="stretch", height=320)


def render_live_briefing_tab(analysis):
    data = analysis["live_briefing"]
    benchmark_note = build_benchmark_sensitivity_text(analysis)
    summary_cols = st.columns(6)
    brief_metrics = [
        ("1D", format_percent(data["ret_1d"])),
        ("1W", format_percent(data["ret_1w"])),
        ("1M", format_percent(data["ret_1m"])),
        ("YTD", format_percent(data["ytd_ret"])),
        ("Vol 20d", "-" if pd.isna(data["realized_vol_20"]) else f"{data['realized_vol_20']:.1f}%"),
        ("Vol Ratio", "-" if pd.isna(data["volume_ratio"]) else f"{data['volume_ratio']:.2f}x"),
    ]
    for column, (label, value) in zip(summary_cols, brief_metrics):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{label}</div>
                    <div class="kpi-value" style="font-size:1.15rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.45, 1])
    with left:
        close = data["daily"]["Close"].dropna()
        price_df = pd.DataFrame({"Date": close.index, "Close": close.values}).reset_index(drop=True)
        st.markdown('<div class="section-label">1Y Structure</div>', unsafe_allow_html=True)
        st.altair_chart(build_line_area_chart(price_df, "Date", "Close", color="#FFFFFF", fill="#F59E0B", title=None, minimal=False), width="stretch")
    with right:
        st.markdown('<div class="section-label">Research Brief</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="commentary-card">{wrap_text(build_live_briefing_text(data))}</div>', unsafe_allow_html=True)
        if benchmark_note:
            st.markdown('<div class="section-label">Index Sensitivity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="commentary-card">{wrap_text(benchmark_note)}</div>', unsafe_allow_html=True)


def render_secondary_metrics(analysis):
    metrics = analysis["latest_metrics"]
    cols = st.columns(4)
    items = [
        ("RSI 14", f"{metrics['latest_rsi']:.2f}"),
        ("Sharpe 60d", f"{metrics['latest_sharpe']:.2f}"),
        (f"Beta με βαση τον {analysis['benchmark']}", f"{metrics['latest_beta_benchmark']:.2f}"),
        ("Drawdown", f"{metrics['latest_drawdown']:.2f}%"),
    ]
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="kpi-title">{label}</div>
                    <div class="kpi-value" style="font-size:1.12rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_advisory_tab(analysis=None):
    if "logged_in_user" not in st.session_state:
        st.info("Παρακαλώ συνδεθείτε μέσω του Portfolio tab (🔐 Είσοδος) για να δείτε τα Advisory Analytics.")
        return

    user = st.session_state.logged_in_user
    txns = db_get_transactions(user)
    if txns.empty:
        st.info("Προσθέστε συναλλαγές στο Portfolio σας για να δείτε institutional risk analysis.")
        return

    st.markdown('<div class="section-label" style="font-size: 1.2rem; margin-bottom: 0px;">Institutional Advisory & Risk Report</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.9rem; color: var(--muted); margin-bottom: 20px;">Hedge-fund grade risk analytics: CVaR, Harmonic P/E, Equity Risk Premium, Stress Testing.</div>', unsafe_allow_html=True)

    unique_tickers = list(txns["ticker"].unique())
    enrich_metadata(unique_tickers + ["SPY", "QQQ"])

    meta_df_stored = db_get_metadata(unique_tickers + ["SPY", "QQQ"])
    market_data = fetch_market_snapshot(unique_tickers)
    port_df = compute_portfolio_state(txns, market_data)

    if port_df.empty:
        st.info("Δεν βρέθηκαν ενεργές θέσεις.")
        return

    # ── Build sector map for stress testing ──────────────────────────────────
    meta_for_sectors = db_get_metadata(unique_tickers)
    sector_map: dict[str, str] = {}
    if not meta_for_sectors.empty:
        sector_map = meta_for_sectors.set_index("ticker")["sector"].fillna("Unknown").to_dict()

    returns_df = fetch_historical_returns(unique_tickers)
    total_val = port_df["Current Value"].dropna().sum()

    # ── Run Engines ───────────────────────────────────────────────────────────
    quant = calculate_quant_metrics(port_df, returns_df, sector_map=sector_map)
    valuation = calculate_valuation_metrics(port_df, meta_df_stored)

    # ── Section 1: Risk KPIs ──────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="font-size: 1rem; margin-top: 8px;">📊 Risk Engine</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)

    var_95 = quant.get("var_95", np.nan)
    var_95_pct = quant.get("var_95_pct", np.nan)
    cvar_95 = quant.get("cvar_95", np.nan)
    cvar_95_pct = quant.get("cvar_95_pct", np.nan)
    vol = quant.get("portfolio_vol_annual", np.nan)
    rolling_alert = quant.get("rolling_corr_alert", False)

    with r1:
        st.markdown(
            f'<div class="glass-card" title="Το VaR 95% είναι η μέγιστη ζημιά που περιμένετε σε μια κακή ημέρα βάσει 1-έτους ιστορικών δεδομένων."><div class="kpi-title">VaR 95% (1-Day)</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;color:var(--red);">'
            f'{"${:,.0f}".format(var_95) if pd.notna(var_95) else "N/A"}</div>'
            f'<div class="kpi-delta">-{var_95_pct:.2f}%</div></div>'
            if pd.notna(var_95) else
            f'<div class="glass-card"><div class="kpi-title">VaR 95% (1-Day)</div><div class="kpi-value">N/A</div></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="glass-card" title="Το Expected Shortfall. Αν η ημέρα βρεθεί στο ακραίο 5%, το CVaR υπολογίζει τον ΜΕΣΟ όρο αυτής της ακραίας ζημιάς. Μετράει Black Swan Events."><div class="kpi-title">CVaR / Exp. Shortfall</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;color:var(--red);">'
            f'{"${:,.0f}".format(cvar_95) if pd.notna(cvar_95) else "N/A"}</div>'
            f'<div class="kpi-delta">-{cvar_95_pct:.2f}% (worst 5% avg)</div></div>'
            if pd.notna(cvar_95) else
            f'<div class="glass-card"><div class="kpi-title">CVaR / Exp. Shortfall</div><div class="kpi-value">N/A</div></div>',
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f'<div class="glass-card"><div class="kpi-title">Annual Volatility</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;">{vol:.1f}%</div>'
            f'<div class="kpi-delta">Annualised σ × √252</div></div>'
            if pd.notna(vol) else
            f'<div class="glass-card"><div class="kpi-title">Annual Volatility</div><div class="kpi-value">N/A</div></div>',
            unsafe_allow_html=True,
        )
    with r4:
        n_high_corr = len(quant.get("correlation_warnings", []))
        corr_color = "var(--red)" if (n_high_corr > 0 or rolling_alert) else "var(--green)"
        corr_label = f"{n_high_corr} ζεύγη >0.80" if n_high_corr else "Επαρκής διαφοροποίηση"
        rolling_badge = " | Regime Shift!" if rolling_alert else ""
        st.markdown(
            f'<div class="glass-card" title="Αυξημένες συσχετίσεις δηλώνουν ότι σε market events όλο το portfolio θα πέσει ταυτόχρονα."><div class="kpi-title">Correlation Status</div>'
            f'<div class="kpi-value" style="font-size:0.95rem;color:{corr_color};">{corr_label}</div>'
            f'<div class="kpi-delta">{rolling_badge if rolling_badge else "Rolling 30d OK"}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Section 2: Valuation KPIs ─────────────────────────────────────────────
    st.markdown('<div class="section-label" style="font-size: 1rem; margin-top: 20px;">📈 Valuation Engine</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)

    h_pe = valuation.get("harmonic_forward_pe", np.nan)
    a_pe = valuation.get("arithmetic_forward_pe", np.nan)
    spy_pe = valuation.get("spy_forward_pe", np.nan)
    bench_name = valuation.get("benchmark_name", "SPY")
    erp = valuation.get("equity_risk_premium_pct", np.nan)
    ey = valuation.get("earnings_yield_pct", np.nan)
    rf = valuation.get("risk_free_rate_pct", np.nan)
    erp_alert = valuation.get("erp_alert", False)
    bias = valuation.get("pe_bias_pct", np.nan)

    # Override benchmark from live analysis if available
    if analysis and "benchmark_financials" in analysis:
        snap = analysis["benchmark_financials"].get("snapshot", {})
        live_spy_pe = snap.get("forward_pe", np.nan)
        if pd.notna(live_spy_pe):
            spy_pe = live_spy_pe
            bench_name = st.session_state.get("benchmark", bench_name).upper()

    with v1:
        help_text = f"Αρμονικός Μέσος P/E (σωστός). Arithmetic = {a_pe:.1f}x (+{bias:.1f}% υπερεκτίμηση)" if pd.notna(a_pe) and pd.notna(bias) else "Harmonic Mean Forward P/E"
        st.metric("Harmonic Forward P/E", f"{h_pe:.1f}x" if pd.notna(h_pe) else "N/A", help=help_text)
    with v2:
        st.metric(f"{bench_name} Forward P/E", f"{spy_pe:.1f}x" if pd.notna(spy_pe) else "N/A",
                  delta=f"{h_pe - spy_pe:+.1f}x vs portfolio" if (pd.notna(h_pe) and pd.notna(spy_pe)) else None)
    with v3:
        st.metric("Earnings Yield", f"{ey:.2f}%" if pd.notna(ey) else "N/A",
                  help="1 / Forward P/E. Η 'απόδοση κερδών' του χαρτοφυλακίου.")
    with v4:
        erp_color = "🔴" if erp_alert else ""
        st.metric(f"{erp_color} Equity Risk Premium",
                  f"{erp:.2f}%" if pd.notna(erp) else "N/A",
                  delta=f"RF Rate: {rf:.2f}%" if pd.notna(rf) else None,
                  help=f"Διαφορά μεταξύ του Earnings Yield (η αναμενόμενη απόδοση των μετοχών) και των ασφαλών ομολόγων. Μετρικό < 1.5% δηλώνει ότι το αγοραίο ρίσκο των μετοχών σας δεν αποζημιώνεται αρκετά σε σύγκριση με risks-free 3μηνα έντοκα. Σηματοδοτεί πιθανή κακή σχέση ρίσκου-απόδοσης.")

    # ── Section 3: Stress Testing ─────────────────────────────────────────────
    stress = quant.get("stress_results", {})
    if stress:
        st.markdown('<div class="section-label" style="font-size: 1rem; margin-top: 20px;">🧪 Stress Testing / Scenario Analysis</div>', unsafe_allow_html=True)
        stress_cols = st.columns(len(stress))
        for col, (sc_name, sc_data) in zip(stress_cols, stress.items()):
            pnl = sc_data["pnl"]
            pct = sc_data["pct"]
            color = "var(--green)" if pnl >= 0 else "var(--red)"
            short_name = sc_name.split("(")[0].strip()
            with col:
                st.markdown(
                    f'<div class="glass-card" title="{sc_data["description"]}">'
                    f'<div class="kpi-title" style="font-size:0.72rem;">{short_name}</div>'
                    f'<div class="kpi-value" style="font-size:1rem;color:{color};">'
                    f'{"${:+,.0f}".format(pnl)}</div>'
                    f'<div class="kpi-delta">{pct:+.1f}%</div></div>',
                    unsafe_allow_html=True,
                )

    # ── INTERACTIVE MACRO SCENARIO ENGINE ─────────────────────────────────────
    st.markdown('<div class="section-label" style="font-size: 1rem; margin-top: 30px;">🌍 Interactive Macro Engine</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: var(--muted); margin-bottom: 10px;">Υπολογισμός σε πραγματικό χρόνο της ζημίας/κέρδους των θέσεών σας βάσει 1y Beta & Correlation όταν το Benchmark μεταβάλλεται.</div>', unsafe_allow_html=True)
    
    from macro_engine import simulate_macro_shock
    import altair as alt
    
    bench_ticker = st.session_state.get("benchmark", "SPY").upper()
    shock_val = st.slider(f"Macro Shock Slider: {bench_ticker} Return (%)", min_value=-50, max_value=50, value=-10, step=1)
    
    impact = simulate_macro_shock(port_df, returns_df, benchmark=bench_ticker, shock_pct=shock_val)
    
    if impact and "total_simulated_value" in impact:
        simulated_pnl = impact["total_expected_pl"]
        total_sim = impact["total_simulated_value"]
        pnl_color = "#ef5350" if simulated_pnl < 0 else "#26a69a"
        
        st.markdown(f'<div style="background:var(--background); padding: 15px; border-radius: 6px; border: 1px solid var(--border); margin-bottom: 20px;">'
                    f'<span style="font-size:1.0rem; color:var(--muted);">Simulated Portfolio Value: </span>'
                    f'<span style="font-size:1.3rem; font-weight:700;">${total_sim:,.2f}</span> '
                    f'<span style="color:{pnl_color}; font-weight:600; margin-left:10px;">(Expected Impact: ${simulated_pnl:+,.2f})</span></div>', 
                    unsafe_allow_html=True)
        
        chart_data = []
        for t, data in impact["asset_impacts"].items():
            if data["current_value"] > 0:
                chart_data.append({
                    "Asset": t,
                    "P/L ($)": data["pl_impact"],
                    "Drop (%)": data["implied_shock_pct"],
                    "Sim. Value": data["simulated_value"]
                })
                
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("P/L ($):Q", title="Portfolio Impact ($)"),
                y=alt.Y("Asset:N", sort=alt.EncodingSortField(field="P/L ($)", order="ascending"), title=None),
                color=alt.condition(alt.datum['P/L ($)'] > 0, alt.value('#26a69a'), alt.value('#ef5350')),
                tooltip=["Asset", alt.Tooltip("P/L ($):Q", format="$,.2f"), alt.Tooltip("Drop (%):Q", format=".2f")]
            ).properties(height=max(150, len(chart_data) * 35))
            st.altair_chart(bars, use_container_width=True)

    # ── Section 4: Rolling Correlation Details ────────────────────────────────
    rolling_summary = quant.get("rolling_corr_current", {})
    if rolling_summary:
        with st.expander("🔄 Rolling 30-Day Correlation Details", expanded=rolling_alert):
            rc_data = [
                {"Ζεύγος": k, "30d Corr": v["current_30d"], "Ιστορικό Avg": v["historical_avg"],
                 "Δ": round(v["current_30d"] - v["historical_avg"], 3)}
                for k, v in rolling_summary.items()
            ]
            rc_df = pd.DataFrame(rc_data).sort_values("30d Corr", ascending=False)
            st.dataframe(rc_df, hide_index=True, use_container_width=True)

    # ── Section 5: Senior Analyst Advisory ───────────────────────────────────
    st.markdown('<div class="section-label" style="font-size: 1.1rem; margin-top: 30px;">💼 Senior Analyst Advisory</div>', unsafe_allow_html=True)
    insights = generate_advisory_insights(quant, valuation, total_val)
    if insights:
        for insight in insights:
            st.info(insight, icon="💡")
    else:
        st.markdown("Δεν υπάρχουν triggers αυτή τη στιγμή για το χαρτοφυλάκιό σας.")



def main():
    st.set_page_config(page_title="Enhanced Quant Analytics", page_icon=":bar_chart:", layout="wide")
    inject_styles()
    init_portfolio_db()

    with st.sidebar:
        st.markdown("### Controls")
        ticker = st.text_input("Ticker", value="NVDA").strip().upper() or "NVDA"
        benchmark = st.text_input("Benchmark", value="QQQ").strip().upper() or "QQQ"
        beta_window = int(st.number_input("Rolling Beta Window", min_value=20, max_value=252, value=60, step=5))
        run = st.button("Run Dashboard", width="stretch")
        st.markdown("---")
        st.caption("Σημείωση: το Tremor είναι React-first library, οπότε εδώ αποδόθηκε μέσω Streamlit με Tailwind/Tremor-inspired custom CSS.")

    if "run_dashboard" not in st.session_state:
        st.session_state.run_dashboard = True

    if run:
        st.session_state.run_dashboard = True
        st.session_state.ticker = ticker
        st.session_state.benchmark = benchmark
        st.session_state.beta_window = beta_window

    ticker = st.session_state.get("ticker", ticker)
    benchmark = st.session_state.get("benchmark", benchmark)
    beta_window = st.session_state.get("beta_window", beta_window)

    render_hero(ticker, benchmark)

    if not st.session_state.run_dashboard:
        st.stop()

    with st.spinner(f"Loading live market, multiple and financial data for {ticker}..."):
        try:
            analysis = build_analysis(ticker, benchmark, beta_window)
        except ValueError as e:
            st.error(f"Σφάλμα δεδομένων: {e}")
            st.warning("Παρακαλώ δοκιμάστε ξανά σε λίγο ή ελέγξτε αν τα Tickers είναι σωστά.")
            st.stop()
        except Exception:
            st.error("Απρόσμενο σφάλμα κατά την επικοινωνία με το Data Feed.")
            st.stop()

    left, right = st.columns([0.78, 2.2], gap="large")
    with left:
        render_kpi_rail(analysis["valuation_cards"])
        render_benchmark_multiples(benchmark, analysis["benchmark_financials"])
        render_analyst_panel()
    with right:
        render_secondary_metrics(analysis)
        tabs = st.tabs(["Overview", "Market Pulse", "Financials", "Live Briefing", "📂 Portfolio", "Advisory & Risk"])
        with tabs[0]:
            render_overview_tab(analysis)
        with tabs[1]:
            render_market_pulse_tab(ticker)
        with tabs[2]:
            render_financials_tab(analysis)
        with tabs[3]:
            render_live_briefing_tab(analysis)
        with tabs[4]:
            render_portfolio_tab()
        with tabs[5]:
            render_advisory_tab(analysis)

if __name__ == "__main__":
    main()
