import pandas as pd
import numpy as np
import datetime


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
        #print(f"{num_spikes}: {max_idx} {max_value}")

        # Save the maximum spike (index and value)
        spikes.append((max_idx, max_value))
        
        # Suppress the values around the maximum spike
        start_idx = max(min_seridx, max_idx - distance)
        end_idx = min(max_seridx, max_idx + distance)
        
        series_copy[start_idx:end_idx] = np.nan
    print(spikes)

    # Convert the result to a DataFrame for better readability
    spikes = list(zip(*spikes))
    series = pd.Series(spikes[1], index=spikes[0])

    return series

def calc_atr_spikes(yf_df:pd.DataFrame, periods=100):

    # we remove the excess data points before the actual AOI that we want.
    stock_df = yf_df[-periods:].copy()

    # create intermediate
    stock_df['ClosePrev'] = stock_df['Close'].shift(1)

    # calc volas
    stock_df['TR'] = stock_df.apply(true_range, axis=1)
    stock_df['LR'] = stock_df.apply(low_range, axis=1)

    # clean up
    stock_df.drop(
        ['ClosePrev'],
        axis=1,
        inplace=True,
        errors='ignore',
    )

    ##---- Full ATR

    outliers = 2
    #outliers = int(len(stock_df) * 0.05)
    atr_markers = stock_df['TR'].sort_values(ascending=False)[:outliers]

    ##---- Neg. ATR

    low_atr_markers = stock_df['LR'].sort_values(ascending=False)[:outliers]

    ##---- Isolated Spikes


    iso_low_atr_markers = find_isolated_spikes(stock_df['LR'], num_spikes=outliers, n_distance=2)

    ##---- Different means
    scaler = 1.0

    markers_series = iso_low_atr_markers

    arithmetic_mean = markers_series.sum() / len(markers_series) * scaler
    harmonic_mean = len(markers_series) / (1 / markers_series).sum() * scaler
    geometric_mean = np.exp(np.log(markers_series).mean()) * scaler


    print(f"""
    {'Arithmetic mean' : <16}: {arithmetic_mean:8.3}
    {'Harmonic mean' : <16}: {harmonic_mean:8.3}
    {'Geometric mean' : <16}: {geometric_mean:8.3}
    """)

    return harmonic_mean
