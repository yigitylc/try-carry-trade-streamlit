"""
data.py
=======
Data fetching, caching, and regime slicing for USD/TRY data.

Features:
- yfinance fetch with auto_adjust=True
- File-based caching to handle throttling
- Demo dataset fallback if yfinance fails
- Regime window slicing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Cache settings
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "usdtry_data.parquet"
CACHE_META = CACHE_DIR / "cache_meta.json"
CACHE_EXPIRY_HOURS = 4

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance not installed. Using demo data.")


def get_demo_data() -> pd.DataFrame:
    """
    Generate synthetic demo data for when yfinance is unavailable.
    Mimics USD/TRY behavior with realistic drift and volatility.
    """
    np.random.seed(42)

    # Generate 3 years of daily data
    end_date = datetime(2025, 12, 20)
    start_date = end_date - timedelta(days=1095)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Parameters mimicking TRY depreciation
    n_days = len(dates)
    mu_daily = 0.0007  # ~18% annual drift
    sigma_daily = 0.0075  # ~12% annual vol

    # Start around 18 TRY/USD 3 years ago
    spot_start = 18.0

    # Generate GBM path
    log_returns = np.random.normal(mu_daily, sigma_daily, n_days)
    log_prices = np.log(spot_start) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    # Scale to end around current levels (~42-43)
    scale_factor = 42.72 / prices[-1]
    prices = prices * scale_factor

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.01, n_days)),
        'Low': prices * (1 - np.random.uniform(0, 0.01, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
    })
    df.set_index('Date', inplace=True)

    return df


def is_cache_valid() -> bool:
    """Check if cache exists and is not expired."""
    if not CACHE_FILE.exists() or not CACHE_META.exists():
        return False

    try:
        with open(CACHE_META, 'r') as f:
            meta = json.load(f)

        cached_time = datetime.fromisoformat(meta['timestamp'])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600

        return age_hours < CACHE_EXPIRY_HOURS
    except:
        return False


def save_to_cache(df: pd.DataFrame):
    """Save data to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_FILE)

        meta = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'start': str(df.index[0]),
            'end': str(df.index[-1]),
        }
        with open(CACHE_META, 'w') as f:
            json.dump(meta, f)

        return True
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")
        return False


def load_from_cache() -> pd.DataFrame:
    """Load data from cache."""
    try:
        df = pd.read_parquet(CACHE_FILE)
        return df
    except Exception as e:
        print(f"[WARNING] Could not load cache: {e}")
        return None


def fetch_usdtry_data(
    period: str = "max",
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch USD/TRY daily data from yfinance.

    Args:
        period: Data period ("1y", "2y", "3y", "5y", "max")
        use_cache: Whether to use file cache
        force_refresh: Force fetch even if cache is valid

    Returns:
        DataFrame with OHLCV data, DatetimeIndex
    """
    # Try cache first (but skip for max if cached data is small)
    if use_cache and not force_refresh and is_cache_valid():
        df = load_from_cache()
        if df is not None:
            # If requesting max but cache has less than 10 years, refresh
            if period.lower() == "max" and len(df) < 2500:
                print(f"[INFO] Cache has {len(df)} rows, fetching max for more data...")
            else:
                print(f"[INFO] Loaded {len(df)} rows from cache")
                return df

    # Try yfinance
    if YFINANCE_AVAILABLE:
        try:
            print(f"[INFO] Fetching USD/TRY data from yfinance (period={period})...")
            ticker = yf.Ticker("USDTRY=X")
            df = ticker.history(period=period, auto_adjust=True)

            if df is not None and len(df) > 0:
                # Clean up timezone
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Cache the data
                if use_cache:
                    save_to_cache(df)

                print(f"[INFO] Fetched {len(df)} rows from yfinance")
                return df
            else:
                print("[WARNING] yfinance returned empty data")

        except Exception as e:
            print(f"[WARNING] yfinance fetch failed: {e}")

    # Try cached data even if expired
    if use_cache:
        df = load_from_cache()
        if df is not None:
            print(f"[WARNING] Using expired cache ({len(df)} rows)")
            return df

    # Fallback to demo data
    print("[INFO] Using demo data (synthetic USD/TRY)")
    return get_demo_data()


def get_spot_series(df: pd.DataFrame = None) -> pd.Series:
    """
    Extract closing price series from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame (fetches if None)

    Returns:
        Series with DatetimeIndex -> Close price
    """
    if df is None:
        df = fetch_usdtry_data()

    return df['Close']


def slice_regime(
    spot_series: pd.Series,
    regime_days: int = None
) -> pd.Series:
    """
    Slice spot series to regime window.

    Args:
        spot_series: Full spot series
        regime_days: Number of calendar days (None = full)

    Returns:
        Sliced series
    """
    if regime_days is None:
        return spot_series

    cutoff = spot_series.index[-1] - pd.Timedelta(days=regime_days)
    return spot_series[spot_series.index >= cutoff]


def get_spot_at_date(
    spot_series: pd.Series,
    target_date: str,
    tolerance_days: int = 5
) -> tuple:
    """
    Get spot rate at or near a target date.

    Args:
        spot_series: Spot series
        target_date: Target date string (YYYY-MM-DD)
        tolerance_days: How many days to search around target

    Returns:
        Tuple of (actual_date, spot_value)
    """
    target = pd.Timestamp(target_date)

    # Try exact match first
    if target in spot_series.index:
        return target, spot_series[target]

    # Search nearby
    for delta in range(1, tolerance_days + 1):
        for direction in [-1, 1]:
            check_date = target + pd.Timedelta(days=delta * direction)
            if check_date in spot_series.index:
                return check_date, spot_series[check_date]

    # Find nearest available
    available = spot_series.index
    idx = available.get_indexer([target], method='nearest')[0]
    if idx >= 0:
        nearest_date = available[idx]
        return nearest_date, spot_series[nearest_date]

    return None, None


def get_regime_options() -> dict:
    """
    Get available regime options.

    Returns:
        Dict mapping display name -> days (None for MAX)
    """
    return {
        '1 Year': 365,
        '2 Years': 730,
        '3 Years': 1095,
        '4 Years': 1460,
        '5 Years': 1825,
        'Max Available': None,
    }


def validate_data_sufficiency(
    spot_series: pd.Series,
    regime_days: int,
    min_returns: int = 60
) -> dict:
    """
    Check if data is sufficient for analysis.

    Args:
        spot_series: Spot series
        regime_days: Requested regime
        min_returns: Minimum required returns

    Returns:
        Dict with validation results and suggested fallback
    """
    sliced = slice_regime(spot_series, regime_days)
    n_points = len(sliced)
    n_returns = n_points - 1

    result = {
        'requested_regime': regime_days,
        'n_points': n_points,
        'n_returns': n_returns,
        'sufficient': n_returns >= min_returns,
        'fallback_regime': None,
    }

    if not result['sufficient']:
        # Find fallback
        fallbacks = [730, 1095, 1825, None]  # 2Y, 3Y, 5Y, MAX
        for fb in fallbacks:
            if fb == regime_days:
                continue
            fb_sliced = slice_regime(spot_series, fb)
            if len(fb_sliced) - 1 >= min_returns:
                result['fallback_regime'] = fb
                break

    return result


def get_data_summary(df: pd.DataFrame = None) -> dict:
    """
    Get summary statistics about available data.

    Args:
        df: OHLCV DataFrame (fetches if None)

    Returns:
        Dict with data summary
    """
    if df is None:
        df = fetch_usdtry_data()

    spot = df['Close']

    return {
        'start_date': spot.index[0].strftime('%Y-%m-%d'),
        'end_date': spot.index[-1].strftime('%Y-%m-%d'),
        'n_observations': len(spot),
        'current_spot': spot.iloc[-1],
        'min_spot': spot.min(),
        'max_spot': spot.max(),
        'mean_spot': spot.mean(),
        '1y_start': spot.iloc[-min(252, len(spot))],
        '1y_return_pct': (spot.iloc[-1] / spot.iloc[-min(252, len(spot))] - 1) * 100 if len(spot) >= 252 else None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DATA MODULE TEST")
    print("=" * 60)

    # Fetch data
    df = fetch_usdtry_data(period="5y")

    # Summary
    summary = get_data_summary(df)
    print("\nData Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Get spot at entry date
    spot_series = get_spot_series(df)
    date, spot = get_spot_at_date(spot_series, "2025-12-17")
    print(f"\nSpot at entry (2025-12-17 or nearest): {date} -> {spot:.4f}")

    # Test regime validation
    for regime in [365, 730, 1095, 1825, None]:
        validation = validate_data_sufficiency(spot_series, regime)
        print(f"\nRegime {regime} days:")
        print(f"  Points: {validation['n_points']}, Returns: {validation['n_returns']}")
        print(f"  Sufficient: {validation['sufficient']}")
        if validation['fallback_regime']:
            print(f"  Fallback: {validation['fallback_regime']} days")
