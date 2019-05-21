from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import classify, shift, create_model, load_data

class PricePrediction:
    """A Class utility to train and predict price of stocks/cryptocurrencies/trades
        using keras model"""
    def __init__(self, ticker_name, **kwargs):
        """
        :param ticker_name (str): ticker name, e.g. aapl, nflx, etc.
        :param n_steps (int): sequence length used to predict, default is 60
        :param price_column (str): the name of column that contains price predicted, default is 'adjclose'
        :param feature_columns (list): a list of feature column names used to train the model, 
            default is ['adjclose', 'volume', 'open', 'high', 'low']
        :param target_column (str): target column name, default is 'future'
        :param lookup_step (int): the future lookup step to predict, default is 1 (e.g. next day)
        :param shuffle (bool): whether to shuffle the dataset, default is True
        :param verbose (int): verbosity level, default is 1
        ==========================================
        Model parameters
        :param n_layers (int): number of recurrent neural network layers, default is 3
        :param cell (keras.layers.RNN): RNN cell used to train keras model, default is LSTM
        :param units (int): number of units of `cell`, default is 256
        :param dropout (float): dropout rate ( from 0 to 1 ), default is 0.3
        ==========================================
        Training parameters
        :param batch_size (int): number of samples per gradient update, default is 64
        :param epochs (int): number of epochs, default is 100
        :param optimizer (str, keras.optimizers.Optimizer): optimizer used to train, default is 'adam'
        :param loss (str, function): loss function used to minimize during training,
            default is 'mae'
        :param test_size (float): test size ratio from 0 to 1, default is 0.15
        """
        self.ticker_name = ticker_name
        self.n_steps = kwargs.get("n_steps", 60)
        self.price_column = kwargs.get("price_column", 'adjclose')
        self.feature_columns = kwargs.get("feature_columns", ['adjclose', 'volume', 'open', 'high', 'low'])
        self.target_column = kwargs.get("target_column", "future")
        self.lookup_step = kwargs.get("lookup_step", 1)
        self.shuffle = kwargs.get("shuffle", True)
        self.verbose = kwargs.get("verbose", 1)

        self.n_layers = kwargs.get("n_layers", 3)
        self.cell = kwargs.get("cell", LSTM)
        self.units = kwargs.get("units", 256)
        self.dropout = kwargs.get("dropout", 0.3)

        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "mae")
        self.test_size = kwargs.get("test_size", 0.15)

        # create unique model name
        self._update_model_name()

        # runtime attributes
        self.model_trained = False
        self.data_loaded = False
        self.model_created = False

        # test price values
        self.test_prices = None
        # predicted price values for the test set
        self.y_pred = None

        # prices converted to buy/sell classes
        self.classified_y_true = None
        # predicted prices converted to buy/sell classes
        self.classified_y_pred = None

        # most recent price
        self.last_price = None

        # make folders if does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        if not os.path.isdir("data"):
            os.mkdir("data")

    def create_model(self):
        """Construct and compile the keras model"""
        self.model = create_model(input_length=self.n_steps,
                                    units=self.units,
                                    cell=self.cell,
                                    dropout=self.dropout,
                                    n_layers=self.n_layers,
                                    loss=self.loss,
                                    optimizer=self.optimizer)
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def train(self, override=False):
        """Train the keras model using `self.checkpointer` and `self.tensorboard` as keras callbacks.
        If model created already trained, this method will load the weights instead of training from scratch.
        Note that this method will create the model and load data if not called before."""
        
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if data isn't loaded yet, load it
        if not self.data_loaded:
            self.load_data()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=f"logs/{self.model_name}")

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard])
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, classify=False):
        """Predicts next price for the step `self.lookup_step`.
            when `classify` is True, returns 0 for sell and 1 for buy"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        # reshape to fit the model input
        last_sequence = self.last_sequence.reshape((self.last_sequence.shape[1], self.last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        predicted_price = self.column_scaler[self.price_column].inverse_transform(self.model.predict(last_sequence))[0][0]
        if classify:
            last_price = self.get_last_price()
            return 1 if last_price < predicted_price else 0
        else:
            return predicted_price

    def load_data(self):
        """Loads and preprocess data"""
        filename, exists = self._df_exists()
        if exists:
            # if the updated dataframe already exists in disk, load it
            self.ticker = pd.read_csv(filename)
            ticker = self.ticker
            if self.verbose > 0:
                print("[*] Dataframe loaded from disk")
        else:
            ticker = self.ticker_name

        result = load_data(ticker,n_steps=self.n_steps, lookup_step=self.lookup_step,
                            shuffle=self.shuffle, feature_columns=self.feature_columns,
                            price_column=self.price_column, test_size=self.test_size)
        
        # extract data
        self.df = result['df']
        self.X_train = result['X_train']
        self.X_test = result['X_test']
        self.y_train = result['y_train']
        self.y_test = result['y_test']
        self.column_scaler = result['column_scaler']
        self.last_sequence = result['last_sequence']      

        if self.shuffle:
            self.unshuffled_X_test = result['unshuffled_X_test']
            self.unshuffled_y_test = result['unshuffled_y_test']
        else:
            self.unshuffled_X_test = self.X_test
            self.unshuffled_y_test = self.y_test

        self.original_X_test = self.unshuffled_X_test.reshape((self.unshuffled_X_test.shape[0], self.unshuffled_X_test.shape[2], -1))
        
        self.data_loaded = True
        if self.verbose > 0:
            print("[+] Data loaded")

        # save the dataframe to disk
        self.save_data()

    def get_last_price(self):
        """Returns the last price ( i.e the most recent price )"""
        if not self.last_price:
            self.last_price = float(self.df[self.price_column].tail(1))
        return self.last_price

    def get_test_prices(self):
        """Returns test prices. Note that this function won't return the whole sequences,
        instead, it'll return only the last value of each sequence"""
        if self.test_prices is None:
            current = np.squeeze(self.column_scaler[self.price_column].inverse_transform([[ v[-1][0] for v in self.original_X_test ]]))
            future = np.squeeze(self.column_scaler[self.price_column].inverse_transform(np.expand_dims(self.unshuffled_y_test, axis=0)))
            self.test_prices = np.array(list(current) + [future[-1]])
        return self.test_prices

    def get_y_pred(self):
        """Get predicted values of the testing set of sequences ( y_pred )"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        if self.y_pred is None:
            self.y_pred = np.squeeze(self.column_scaler[self.price_column].inverse_transform(self.model.predict(self.unshuffled_X_test)))
        return self.y_pred

    def get_y_true(self):
        """Returns original `y` testing values ( y_true )"""
        test_prices = self.get_test_prices()
        return test_prices[1:]

    def _get_shifted_y_true(self):
        """Returns original `y` testing values shifted by -1.
        This function is useful for converting to a classification problem"""
        test_prices = self.get_test_prices()
        return test_prices[:-1]

    def _calc_classified_prices(self):
        """Convert regression predictions to a classification predictions ( buy or sell )
        and set results to `self.classified_y_pred` for predictions and `self.classified_y_true` 
        for true prices"""
        if self.classified_y_true is None or self.classified_y_pred is None:
            current_prices = self._get_shifted_y_true()
            future_prices = self.get_y_true()
            predicted_prices = self.get_y_pred()
            self.classified_y_true = list(map(classify, current_prices, future_prices))
            self.classified_y_pred = list(map(classify, current_prices, predicted_prices))
        
    # some metrics

    def get_MAE(self):
        """Calculates the Mean-Absolute-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_absolute_error(y_true, y_pred)

    def get_MSE(self):
        """Calculates the Mean-Squared-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_squared_error(y_true, y_pred)

    def get_accuracy(self):
        """Calculates the accuracy after adding classification approach (buy/sell)"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        self._calc_classified_prices()
        return accuracy_score(self.classified_y_true, self.classified_y_pred)

    def plot_test_set(self):
        """Plots test data"""
        future_prices = self.get_y_true()
        predicted_prices = self.get_y_pred()
        plt.plot(future_prices, c='b')
        plt.plot(predicted_prices, c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def save_data(self):
        """Saves the updated dataframe if it does not exist"""
        filename, exists = self._df_exists()
        if not exists:
            self.df.to_csv(filename)
            if self.verbose > 0:
                print("[+] Dataframe saved")

    def _update_model_name(self):
        stock = self.ticker_name.replace(" ", "_")
        feature_columns_str = ''.join([ c[0] for c in self.feature_columns ])
        time_now = time.strftime("%Y-%m-%d")
        self.model_name = f"{time_now}_{stock}-{feature_columns_str}-loss-{self.loss}-{self.cell.__name__}-seq-{self.n_steps}-step-{self.lookup_step}-layers-{self.n_layers}-units-{self.units}"

    def _get_df_name(self):
        """Returns the updated dataframe name"""
        time_now = time.strftime("%Y-%m-%d")
        return f"data/{self.ticker_name}_{time_now}.csv"

    def _df_exists(self):
        """Check if the updated dataframe exists in disk, returns a tuple contains (filename, file_exists)"""
        filename = self._get_df_name()
        return filename, os.path.isfile(filename)

    def _get_model_filename(self):
        """Returns the relative path of this model name with `h5` extension"""
        return f"results/{self.model_name}.h5"

    def _model_exists(self):
        """Checks if model already exists in disk, returns the filename,
        returns `None` otherwise"""
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None