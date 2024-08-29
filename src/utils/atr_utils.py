import pandas as pd
import numpy as np
import datetime

import logging

logger = logging.getLogger(__name__)

# calculate true range
def true_range(x):
    high = x['High']
    low = x['Low']
    close_p = x['ClosePrev']

    if pd.isna(close_p):
        return high - low
    else:
        return max(
            high - low,
            abs(high - close_p),
            abs(close_p - low),
        )

# calculate low range
def low_range(x):
    open = x['Open']
    high = x['High']
    low = x['Low']
    close_p = x['ClosePrev']

    if pd.isna(close_p):
        return (open + high) / 2.0 - low
    else:
        return max((close_p + high) / 2.0 - low, close_p - low, 0)

# calculate high range
def high_range(x):
    open = x['Open']
    high = x['High']
    low = x['Low']
    close_p = x['ClosePrev']

    if pd.isna(close_p):
        return high - (open + low) / 2.0
    else:
        return max(high - (close_p + low) / 2.0, high - close_p, 0)

def find_isolated_spikes(series, num_spikes=10, n_distance=5):
    """
    Find isolated maximum spikes in a Pandas series with a safety distance between each event.
    
    :param series: Input Pandas Series.
    :param num_spikes: Number of spikes to identify.
    :param n_distance: Safety distance between each identified spike.
    :return: DataFrame with the positions and values of the identified spikes.
    """

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


def calc_atr_spikes(yf_df:pd.DataFrame, periods=100):

    # We remove the excess data points before the actual AOI that we want.
    stock_df = yf_df[-periods:].copy()

    # Create intermediate
    stock_df['ClosePrev'] = stock_df['Close'].shift(1)

    # Calc volas
    stock_df['HR'] = stock_df.apply(high_range, axis=1)
    stock_df['LR'] = stock_df.apply(low_range, axis=1)

    high_markers_series = find_isolated_spikes(stock_df['HR'], num_spikes=10, n_distance=5)
    low_markers_series = find_isolated_spikes(stock_df['LR'], num_spikes=10, n_distance=5)

    # Merge series to df
    extrema_df = pd.concat([high_markers_series,low_markers_series], axis=1, keys=['HR', 'LR'])
    extrema_df.sort_index(ascending=True, axis=0, inplace=True)

    ##---- Different means
    scaler = 1.0

    result = []
    # Take valid extrema
    for item in ['HR', 'LR']:
        markers_series = extrema_df[~extrema_df[item].isna()][item]

        arithmetic_mean = markers_series.sum() / len(markers_series) * scaler
        harmonic_mean = len(markers_series) / (1 / markers_series).sum() * scaler
        geometric_mean = np.exp(np.log(markers_series).mean()) * scaler
        extrema_ls = extrema_df[item].nlargest(3).to_list()
        logger.debug(f""" {item} extrema:
        {'Arithmetic mean' : <16}: {arithmetic_mean:8.2f}
        {'Harmonic mean' : <16}: {harmonic_mean:8.2f}
        {'Geometric mean' : <16}: {geometric_mean:8.2f}
        {'Extrema' : <16}: {extrema_ls}
        """)

        result.append(geometric_mean)
        result += extrema_ls

    logger.info(result)
    return result
