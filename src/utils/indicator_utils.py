import pandas as pd
import numpy as np
import datetime
from math import  log2

import logging

logger = logging.getLogger(__name__)

# scale price action
def calibrate_prices(df, key='Close'):
    logger.info(f"Scaling equity price action data based on first {key}")
    
    columns = ['Close', 'Open', 'High', 'Low', 'Adj Close']
    # Key needs to be in columns
    assert key in columns
    
    # Verify all yf columns exist
    for col in columns:
        assert col in df.columns

    df_scaled = df.copy()
    scaler = df[key].iloc[0]
    for col in columns:
        df_scaled[col] = df_scaled[col] / scaler * 100.0
    return df_scaled

# scale trading volume action
def calibrate_volume(df, key='Volume', method='First', divisor=100000, scaler=100000):
    
    # Key needs to be in columns
    assert key in df.columns

    # Identify scaler
    if method=='First':
        divisor = df[key].iloc[0]
    elif method=='Manual':
        divisor = int(divisor)
    elif method=='Median':
        divisor = df[key].median()
    else:
        method = 'No-Op'
        divisor = 1
        scaler = 1
    assert divisor > 0
    assert scaler > 0

    logger.info(f"Scaling trading volume data based on {key} and method {method} with a divisor of {divisor} and scaler {scaler}")

    df_scaled = df.copy()

    df_scaled[key] = df_scaled[key] / divisor * scaler
    df_scaled = df_scaled.astype({key: int})
    return df_scaled

# calculate sum of maximal price envelopes
def price_movement(df, key_dict={'Open': 'Open', 'Close': 'Close', 'High': 'High', 'Low': 'Low'}, length=14):
    logger.info(f"Calculating accumulated price movement({length})")

    df_copy = df.copy()
    # Verify all yf columns exist
    for key, value in key_dict.items():
        assert value in df_copy.columns

    # shift close
    df_copy['p_close'] = df_copy[key_dict['Close']].shift(1).bfill()

    def max_price(x):
        open = x[key_dict['Open']]
        close = x[key_dict['Close']]
        high = x[key_dict['High']]
        low = x[key_dict['Low']]
        p_close = x['p_close']
        main_comp = abs(open - p_close) + 2 * high - 2 * low
        return max(
            main_comp + close - open,
            main_comp - close + open
        )

    series = df_copy.apply(max_price, axis=1)
    series = series.rolling(length, min_periods=1).sum()
    return series

# calculate simple moving average
def sma(series, length=9):
    logger.info(f"Calculating SMA({length})")
    # omitting min_periods creates gap
    return series.rolling(length, min_periods=1).mean()


# calculate exponential moving average
def ema(series, length=9):
    logger.info(f"Calculating EMA({length})")

    series = series.sort_index()
    # adjust = False enables recursion formula
    return series.ewm(span=length,min_periods=0,adjust=False,ignore_na=False).mean()

# calculate range
def vola_range(close, high, low, mode='ATR', length=9, percentile=False):
    if mode not in ['ATR', 'AHR', 'ALR']:
        mode = 'ATR'

    # shift close
    close_p = close.shift(1).bfill()
    df = pd.concat([close_p, high, low], axis=1)
    df.columns = ['close_p', 'high', 'low']

    if mode == 'ALR':
        logger.info(f"Calculating ALR({length})")
        def atr_function(x):
            close_p = x['close_p']
            high = x['high']
            low = x['low']
            return max(
                (close_p + high) / 2.0 - low,
                close_p - low,
                0
            )
    elif mode == 'AHR':
        logger.info(f"Calculating AHR({length})")
        def atr_function(x):
            close_p = x['close_p']
            high = x['high']
            low = x['low']
            return max(
                high - (close_p + low) / 2.0,
                high - close_p,
                0
            )
    else:
        logger.info(f"Calculating ATR({length})")
        def atr_function(x):
            close_p = x['close_p']
            high = x['high']
            low = x['low']
            return max(
                high - low,
                abs(high - close_p),
                abs(close_p - low),
            )

    series = df.apply(atr_function, axis=1)
    series = series.rolling(length, min_periods=1).mean()
    if percentile:
        return series / close
    return series

# calculate signals
def crossovers(short, long, scaledsignal=False):

    calc_df = pd.concat([short, long], axis=1)
    calc_df.columns = ['s', 'l']

    # Get scaler
    sig_str = ''
    sig_scale = 0
    if scaledsignal:
        sig_str = 'scaled'
        sig_scale = 2**int(log2(calc_df['s'].min()))

    logger.info(f"Calculating {sig_str} mode ({sig_scale}) and trading signals")


    # Get relation of curves (short curves higher than long curve: positive (LONG/BUY))
    calc_df['m'] = np.where(short >= long,sig_scale + 1, 0)
    calc_df['m'] = np.where(short < long, sig_scale - 1, calc_df['m'])

    # Signals are mode changes, i.e., crossovers
    calc_df['p'] = np.where(
        np.logical_and(
            calc_df['m'] > sig_scale,
            calc_df['m'].shift(1) < sig_scale,
        ),
        sig_scale,
        None
    )

    calc_df['n'] = np.where(
        np.logical_and(
            calc_df['m'] < sig_scale,
            calc_df['m'].shift(1) > sig_scale,
        ),
        sig_scale,
        None
    )
    calc_df = calc_df[['m', 'p', 'n']]
    calc_df.columns = ['Mode', 'P_Signal', 'N_Signal']

    return calc_df


# Calculate Wilder's relative strength index
def wilder_rsi(close, open=None, length=14):
    logger.info(f"Calculating RSI({length})")
    # Gains/losses during trading hours
    gains = close.diff(1)
    if open is not None:
        gains = (close - open) / open
    
    df = pd.concat([close, gains], axis=1)
    df.columns = ['close', 'diff']
    df = df.fillna(0.0)

    # Split gains and losses
    df['gain'] = df['diff'].clip(lower=0)
    df['loss'] = df['diff'].clip(upper=0).abs()


    for col in ['gain', 'loss']:
        # Calculate initial average gains and losses using rolling mean
        df[f"avg_{col}"] = df[col].rolling(window=length, min_periods=1).mean()
        s_col = df.columns.get_loc(col)
        t_col = df.columns.get_loc(f"avg_{col}")
        # Apply Wilder's smoothing formula using apply after the initial window
        for i in range(length + 1, len(df)):
            df.iloc[i, t_col] = (df.iloc[i - 1, t_col] * (length - 1) + df.iloc[i, s_col]) / length


    # Calculate RS and RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))
    df = df.fillna(0)
    series = df['rsi']
    series[:5] = 50.0

    return series


def find_isolated_spikes(series, num_spikes=10, n_distance=5):
    """
    Find isolated maximum spikes in a Pandas series with a safety distance between each
    event.

    :param series: Input Pandas Series.
    :param num_spikes: Number of spikes to identify.
    :param n_distance: Safety distance between each identified spike.
    :return: DataFrame with the positions and values of the identified spikes.
    """
    logger.info(f"Identifying {num_spikes} isolated spikes ({n_distance})")

    # Copy the series to avoid modifying the original
    series_copy = series.copy()

    # List to hold the positions and values of the identified spikes
    spikes = []

    distance = n_distance * datetime.timedelta(days=1)

    max_seridx = max(series.index)
    min_seridx = min(series.index)

    for _ in range(num_spikes):
        # Find the index of the maximum value in the series
        max_idx = series_copy.idxmax()
        max_value = series_copy[max_idx]
        logger.debug(f"{num_spikes}: {max_idx} {max_value}")

        # Save the maximum spike (index and value)
        spikes.append((max_idx, max_value))

        # Suppress the values around the maximum spike
        start_idx = max(min_seridx, max_idx - distance)
        end_idx = min(max_seridx, max_idx + distance)

        series_copy[start_idx:end_idx] = np.nan

    # Convert the result to a DataFrame for better readability
    spikes = list(zip(*spikes))
    series = pd.Series(spikes[1], index=spikes[0])

    return series


def calc_atr_spikes(yf_df: pd.DataFrame, clip_periods=100, length=1, num_spikes=10, n_distance=5):
    """
    Convenience function to extract ALR/AHR extrema.

    :param yf_df: Input Yahoo Finance dataframe.
    :param clip_periods: input data clipper (keep last n periods).
    :param length: pre-extraction smoothing factor.
    :param num_spikes: Number of spikes to identify.
    :param n_distance: Safety distance between each identified spike.
    :return: List of extrema
    """

    logger.info(f"Getting stats for {num_spikes} isolated HR/LR spikes ({n_distance})")

    for col in ['Close', 'High', 'Low']:
        assert col in yf_df.columns

    # We remove the excess data points before the actual AOI that we want.
    stock_df = yf_df[-clip_periods:].copy()

    # Calc volas
    for opt in ['LR', 'HR']:
        stock_df[opt] = vola_range(
            close=stock_df['Close'],
            high=stock_df['High'],
            low=stock_df['Low'],
            mode=f"A{opt}",
            length=length
        )

    high_markers_series = find_isolated_spikes(
        stock_df['HR'], num_spikes=num_spikes, n_distance=n_distance
    )
    low_markers_series = find_isolated_spikes(
        stock_df['LR'], num_spikes=num_spikes, n_distance=n_distance
    )

    # Merge series to df
    extrema_df = pd.concat(
        [high_markers_series, low_markers_series], axis=1, keys=['HR', 'LR']
    )
    extrema_df.sort_index(ascending=True, axis=0, inplace=True)

    scaler = 1.0

    result = []
    # Take valid extrema
    for item in ['HR', 'LR']:
        markers_series = extrema_df[~extrema_df[item].isna()][item]

        arithmetic_mean = markers_series.sum() / len(markers_series) * scaler
        harmonic_mean = len(markers_series) / (1 / markers_series).sum() * scaler
        geometric_mean = np.exp(np.log(markers_series).mean()) * scaler
        extrema_ls = extrema_df[item].nlargest(3).to_list()
        logger.debug(
            f""" {item} extrema:
        {'Arithmetic mean' : <16}: {arithmetic_mean:8.2f}
        {'Harmonic mean' : <16}: {harmonic_mean:8.2f}
        {'Geometric mean' : <16}: {geometric_mean:8.2f}
        {'Extrema' : <16}: {extrema_ls}
        """
        )

        result.append(geometric_mean)
        result += extrema_ls

    logger.info(result)
    return result
