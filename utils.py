from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from sklearn import preprocessing
from yahoo_fin import stock_info as si
from collections import deque

import pandas as pd
import numpy as np
import random

def create_model(input_length, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
            model.add(Dropout(dropout))
        elif i == n_layers -1:
            # last layer
            model.add(cell(units, return_sequences=False))
            model.add(Dropout(dropout))
        else:
            # middle layers
            model.add(cell(units, return_sequences=True))
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        
    return model


def load_data(ticker, n_steps=60, scale=True, split=True, balance=False, shuffle=True,
                lookup_step=1, test_size=0.15, price_column='Price', feature_columns=['Price'],
                target_column="future", buy_sell=False):
    """Loads data from yahoo finance, if the ticker is a pd Dataframe,
    it'll use it instead"""
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str, or a `pd.DataFrame` instance")

    result = {}

    result['df'] = df.copy()
    # make sure that columns passed is in the dataframe
    for col in feature_columns:
        assert col in df.columns
    
    column_scaler = {}
    if scale:
        # scale the data ( from 0 to 1 )
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # df[column] = preprocessing.scale(df[column].values)

    # add column scaler to the result
    result['column_scaler'] = column_scaler

    # add future price column ( shift by -1 )
    df[target_column] = df[price_column].shift(-lookup_step)

    # get last feature elements ( to add them to the last sequence )
    # before deleted by `df.dropna`
    last_feature_element = np.array(df[feature_columns].tail(1))

    # clean NaN entries
    df.dropna(inplace=True)

    if buy_sell:
        # convert target column to 0 (for sell -down- ) and to 1 ( for buy -up-)
        df[target_column] = list(map(classify, df[price_column], df[target_column]))

    seq_data = [] # all sequences here
    # sequences are made with deque, which keeps the maximum length by popping out older values as new ones come in
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df[target_column].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            seq_data.append([np.array(sequences), target])

    # get the last sequence for future predictions
    last_sequence = np.array(sequences)
    # shift the sequence, one element is missing ( deleted by dropna )
    last_sequence = shift(last_sequence, -1)
    # fill the last element
    last_sequence[-1] = last_feature_element

    # add last sequence to results
    result['last_sequence'] = last_sequence

    if buy_sell and balance:
        buys, sells = [], []
        for seq, target in seq_data:
            if target == 0:
                sells.append([seq, target])
            else:
                buys.append([seq, target])

        # balancing the dataset
        
        lower_length = min(len(buys), len(sells))

        buys = buys[:lower_length]
        sells = sells[:lower_length]

        seq_data = buys + sells

    if shuffle:
        unshuffled_seq_data = seq_data.copy()
        # shuffle data
        random.shuffle(seq_data)

    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        unshuffled_X, unshuffled_y = [], []
        for seq, target in unshuffled_seq_data:
            unshuffled_X.append(seq)
            unshuffled_y.append(target)
        
        unshuffled_X = np.array(unshuffled_X)
        unshuffled_y = np.array(unshuffled_y)

        unshuffled_X = unshuffled_X.reshape((unshuffled_X.shape[0], unshuffled_X.shape[2], unshuffled_X.shape[1]))

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    if not split:
        # return original_df, X, y, column_scaler, last_sequence
        result['X'] = X
        result['y'] = y
        return result
    else:
        # split dataset into training and testing
        n_samples = X.shape[0]
        train_samples = int(n_samples * (1 - test_size))
        result['X_train'] = X[:train_samples]
        result['X_test'] = X[train_samples:]
        result['y_train'] = y[:train_samples]
        result['y_test'] = y[train_samples:]
        if shuffle:
            result['unshuffled_X_test'] = unshuffled_X[train_samples:]
            result['unshuffled_y_test'] = unshuffled_y[train_samples:]
        return result

# from sentdex
def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result