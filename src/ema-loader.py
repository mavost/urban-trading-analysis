#!/usr/bin/env python
# coding: utf-8

# # Notebook to play with EMAs
# 
# Latest version: 2024-08-1620
# Author: MvS
# 
# ## Description
# 
# Similar approach to SMA notebook but using the more volatile exponential moving average (EMA) indicator.
# 
# ## Result
# 
# A vague trading signal (Long/Short) based on Low/High EMAs and a stop-loss estimation.
# 

import yfinance as yf
import pandas as pd

import datetime
import csv
from math import log2


# Standard SMAs
periods = [200, 50]

# parameters of the EMA:
# window length
ema_length = 30
# widening factor
ema_scaler = 1.03

dt_end = datetime.datetime.today()
current_date = dt_end.strftime('%Y-%m-%d')

# Define real-time interval:
#  - assume to display at least the number of sample points of the larger period
#  - this requires double the number of points to create the averaging
#  - plus considering non-trading days - yfinance returns only trading days, howevers
dt_data_start = dt_end - datetime.timedelta(days=max(periods + [ema_length]) * 3)

write_header = True

stocks = [
    ('III', 'Financial services'),
    ('ADM', 'Insurance'),
    ('AAF', 'Telecommunications services'),
    ('AAL', 'Mining'),
    ('ANTO', 'Mining'),
    ('AHT', 'Support services'),
    ('ABF', 'Food & tobacco'),
    ('AZN', 'Pharmaceuticals & biotechnology'),
    ('AUTO', 'Media'),
    ('AV', 'Life insurance'),
    ('BME', 'Retailers'),
    ('BA', 'Aerospace & defence'),
    ('BARC', 'Banks'),
    ('BDEV', 'Household goods & home construction'),
    ('BEZ', 'Insurance'),
    ('BKG', 'Household goods & home construction'),
    ('BP', 'Oil & gas producers'),
    ('BATS', 'Tobacco'),
    ('BT-A', 'Telecommunications services'),
    ('BNZL', 'Support services'),
    ('BRBY', 'Personal goods'),
    ('CNA', 'Multiline utilities'),
    ('CCH', 'Beverages'),
    ('CPG', 'Support services'),
    ('CTEC', 'Health care equipment & supplies'),
    ('CRDA', 'Chemicals'),
    ('DARK', 'Software & Computer Services'),
    ('DCC', 'Support services'),
    ('DGE', 'Beverages'),
    ('DPLM', 'Industrial Support services'),
    ('EDV', 'Precious Metals and Mining'),
    ('ENT', 'Travel & leisure'),
    ('EZJ', 'Travel & leisure'),
    ('EXPN', 'Support services'),
    ('FCIT', 'Financial services'),
    ('FRAS', 'Retailers'),
    ('FRES', 'Mining'),
    ('GLEN', 'Mining'),
    ('GSK', 'Pharmaceuticals & biotechnology'),
    ('HLN', 'Pharmaceuticals & biotechnology'),
    ('HLMA', 'Electronic equipment & parts'),
    ('HL', 'Financial services'),
    ('HIK', 'Pharmaceuticals & biotechnology'),
    ('HWDN', 'Homebuilding & construction supplies'),
    ('HSBA', 'Banks'),
    ('IHG', 'Travel & leisure'),
    ('IMI', 'Machinery, tools, heavy vehicles, trains & ships'),
    ('IMB', 'Tobacco'),
    ('INF', 'Media'),
    ('ICG', 'Financial services'),
    ('IAG', 'Travel & leisure'),
    ('ITRK', 'Support services'),
    ('JD', 'General retailers'),
    ('KGF', 'Retailers'),
    ('LAND', 'Real estate investment trusts'),
    ('LGEN', 'Life insurance'),
    ('LLOY', 'Banks'),
    ('LMP', 'Real Estate Investment Trusts'),
    ('LSEG', 'Financial services'),
    ('MNG', 'Financial services'),
    ('MKS', 'Food & drug retailing'),
    ('MRO', 'Aerospace & defence'),
    ('MNDI', 'Containers & packaging'),
    ('NG', 'Multiline utilities'),
    ('NWG', 'Banks'),
    ('NXT', 'General retailers'),
    ('PSON', 'Media'),
    ('PSH', 'Financial services'),
    ('PSN', 'Household goods & home construction'),
    ('PHNX', 'Life insurance'),
    ('PRU', 'Life insurance'),
    ('RKT', 'Household goods & home construction'),
    ('REL', 'Media'),
    ('RTO', 'Support services'),
    ('RMV', 'Media'),
    ('RIO', 'Mining'),
    ('RR', 'Aerospace & defence'),
    ('SGE', 'Software & computer services'),
    ('SBRY', 'Food & drug retailing'),
    ('SDR', 'Financial services'),
    ('SMT', 'Collective investments'),
    ('SGRO', 'Real estate investment trusts'),
    ('SVT', 'Multiline utilities'),
    ('SHEL', 'Oil & gas producers'),
    ('SMDS', 'General industrials'),
    ('SMIN', 'General industrials'),
    ('SN', 'Health care equipment & supplies'),
    ('SPX', 'Industrial engineering'),
    ('SSE', 'Electrical utilities & independent power producers'),
    ('STAN', 'Banks'),
    ('TW', 'Household goods & home construction'),
    ('TSCO', 'Food & drug retailing'),
    ('ULVR', 'Personal goods'),
    ('UU', 'Multiline utilities'),
    ('UTG', 'Real estate investment trusts'),
    ('VTY', 'Home Construction'),
    ('VOD', 'Mobile telecommunications'),
    ('WEIR', 'Industrial goods and services'),
    ('WTB', 'Retail hospitality'),
    ('WPP', 'Media')
]


for stock, sector in stocks:

    print(
        f"""Getting market data for {stock}."""
    )

    try:
        # Grab sufficient stock data for averaging SMAs
        load_df = yf.download(
            f"{stock}.L",
            start=dt_data_start.strftime('%Y-%m-%d'),
            end=dt_end.strftime('%Y-%m-%d'),
            progress=False,
        )

        assert load_df.shape[1] == 6 and load_df.shape[0] > max(periods + [ema_length])

    except AssertionError:
        print(f"Download failed for symbol {stock}.  Skipping...")
        continue

    # ### Define dynamic stop-losses based on EMAs
    # 
    # - use worst price quote of each day and calculate an exponential 30-day mean to define a stop-loss for long positions
    # - reciprocately, the best price quote of each day and calculate an exponential 30-day mean to define a stop-loss for short positions
    # - both curves are tracking each other with an offset and define a corridor of insignificant price action
    # 

    stock_df = load_df.copy()

    # Compute the simple moving average (SMA)
    for period in periods:
        stock_df[f"SMA_{period:03}"] = stock_df["Close"].rolling(window=period).mean()

    # Compute two scaled EMAs based on daily Highs and Lows
    stock_df[f"EMA_{ema_length:03}_Long"] = (
        stock_df['Low'].ewm(span=ema_length, adjust=False).mean() / ema_scaler
    )  # STOP LOSS LONG
    stock_df[f"EMA_{ema_length:03}_Short"] = (
        stock_df['High'].ewm(span=ema_length, adjust=False).mean() * ema_scaler
    )  # STOP LOSS SHORT

    # Now that we calculated the SMAs and EMAs, we can remove the data points before the actual AOI that we want.
    stock_df = stock_df[-max(periods) :].copy()

    # Add helpers for time period calculation
    dt_start = stock_df.index[0]
    stock_df['dt_start'] = dt_start
    stock_df['dt_end'] = dt_end

    # Add helper to scale signal strength
    sig_scale = int(log2(stock_df['SMA_200'][-ema_length:].mean()))
    stock_df['sig_scale'] = sig_scale

    # Define the corridors for operation as strong deviations of fast from slow SMA
    def valid_signal(x):
        high = f"High"
        low = f"Low"
        long = f"EMA_{ema_length:03}_Long"
        short = f"EMA_{ema_length:03}_Short"

        corridor_scaler = 1.00

        if x[low] > x[short] * corridor_scaler:
            return 1
        elif x[high] < x[long] / corridor_scaler:
            return -1
        else:
            return 0

    stock_df['Valid'] = stock_df.apply(valid_signal, axis=1)

    stock_df['Valid_scaled'] = 2**sig_scale + (stock_df['Valid'] * 2 ** (sig_scale - 2))

    # Despiking a curve for single outliers

    # Compare neighbor values: spike has two opposites on either side with same polarity
    def despike(x):
        if x['Valid_N_Shift'] == x['Valid_P_Shift'] and x['Valid'] != x['Valid_N_Shift']:
            return x['Valid_N_Shift']
        elif pd.isna(x['Valid_P_Shift']):
            return 0.0
        else:
            return x['Valid']

    for iter in range(0, 3):

        # Shifting curve forward / backward
        stock_df['Valid_P_Shift'] = stock_df['Valid'].shift(1)
        stock_df['Valid_N_Shift'] = stock_df['Valid'].shift(-1)

        stock_df['Valid_Despike'] = stock_df.apply(despike, axis=1)

        # Replace old with new
        stock_df['Valid'] = stock_df['Valid_Despike']

    stock_df['Valid_scaled'] = 2**sig_scale + (stock_df['Valid'] * 2 ** (sig_scale - 2))

    # Clean up
    stock_df.drop(
        ['Valid_P_Shift', 'Valid_N_Shift', 'Valid_Despike'],
        axis=1,
        inplace=True,
        errors='ignore',
    )

    # Extract trading signals
    def get_signal(x):
        if x['Valid'] != x['Shift'] and x['Valid'] != 0:
            return x['Valid']
        else:
            return 0

    # Shift forward
    stock_df['Shift'] = stock_df['Valid'].shift(1)
    # Fill NaN
    stock_df['Shift'] = stock_df['Shift'].interpolate(
        method='backfill', limit_direction='backward'
    )

    # Identify signal onsets
    stock_df['On_Signal'] = stock_df.apply(get_signal, axis=1)
    stock_df['On_Signal_scaled'] = 2**sig_scale + (
        stock_df['On_Signal'] * 2 ** (sig_scale - 2)
    )

    # Shift backward
    stock_df['Shift'] = stock_df['Valid'].shift(-1)
    # Fill NaN
    stock_df['Shift'] = stock_df['Shift'].interpolate(
        method='pad', limit_direction='forward'
    )

    # Identify signal terminations
    stock_df['Off_Signal'] = stock_df.apply(get_signal, axis=1)
    stock_df['Off_Signal_scaled'] = 2**sig_scale + (
        stock_df['Off_Signal'] * 2 ** (sig_scale - 2)
    )

    # clean up
    stock_df.drop(
        ['Shift'],
        axis=1,
        inplace=True,
        errors='ignore',
    )

    if stock_df['On_Signal'].any() != 0:
        print('Some signals found...')

        # Filter rows where either 'On_Signal' or 'Off_Signal' is non-zero
        signals_df = stock_df[
            (stock_df['On_Signal'] != 0) | (stock_df['Off_Signal'] != 0)
        ].copy()

        # Calculate length of signals
        signals_df['Signal_len'] = -signals_df.index.to_series().diff(periods=-1).dt.days
        # Calculate age of signals
        signals_df['Signal_age'] = (
            signals_df['dt_end'] - signals_df.index.to_series()
        ).dt.days
        # Add missing length
        signals_df['Signal_len'].fillna(signals_df['Signal_age'], inplace=True)

        # Calculate gap between signals
        signals_df['Signal_gap'] = signals_df.index.to_series().diff(periods=1).dt.days
        # Calculate ref days of signals
        signals_df['Signal_ref'] = (
            signals_df.index.to_series() - signals_df['dt_start']
        ).dt.days
        # Add missing gap
        signals_df['Signal_gap'].fillna(signals_df['Signal_ref'], inplace=True)

    if stock_df['On_Signal'].any() != 0:
        # Add back the signal calculations
        stock_df = pd.concat(
            [
                stock_df,
                signals_df[['Signal_ref', 'Signal_len', 'Signal_gap', 'Signal_age']],
            ],
            axis=1,
        )
        # Fill non-signals
        stock_df.fillna(0, inplace=True)
    else:
        stock_df.loc[:, ['Signal_ref', 'Signal_len', 'Signal_gap', 'Signal_age']] = 0

    # # Type casting
    stock_df = stock_df.astype(
        {
            'Valid': int,
            'Valid_scaled': int,
            'On_Signal': int,
            'On_Signal_scaled': int,
            'Off_Signal': int,
            'Off_Signal_scaled': int,
            'Signal_ref': int,
            'Signal_len': int,
            'Signal_gap': int,
            'Signal_age': int,
        }
    )

    stock_df['Symbol'] = stock
    stock_df['Sector'] = sector

    stock_df['dt_day'] = stock_df.index.strftime('%Y-%m-%d')
    stock_df['dt_end'] = dt_end.strftime('%Y-%m-%d')
    stock_df = stock_df.round(3)

    order = ['Symbol', 'Sector', 'dt_day', 'dt_start', 'dt_end',
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'SMA_200', 'SMA_050', 'EMA_030_Long', 'EMA_030_Short',
        'sig_scale', 'Valid', 'Valid_scaled', 'On_Signal', 'On_Signal_scaled',
        'Off_Signal', 'Off_Signal_scaled' ,
        'Signal_ref', 'Signal_len', 'Signal_gap', 'Signal_age'
    ]

    stock_df[order].to_csv(
        f"../logs/{current_date}_screening.csv",
        sep=",",
        quotechar='"',
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        mode='a',
        header=write_header
    )
    write_header = False
