
# uncomment below to use CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

from keras.layers import GRU, LSTM, CuDNNLSTM
from price_prediction import PricePrediction

ticker = "BTC-USD"

p = PricePrediction("BTC-USD", feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
                    epochs=1000, cell=LSTM, optimizer="adam", n_layers=3, units=256, 
                    loss="mae", shuffle=False)
p.train()
print(f"The next predicted price for {ticker} is {p.predict()}$")
buy_sell = p.predict(classify=True)
print(f"you should {'sell' if buy_sell == 0 else 'buy'}.")

print("Mean Absolute Error:", p.get_MAE())
print("Mean Squared Error:", p.get_MSE())
print(f"Accuracy: {p.get_accuracy()*100:.3f}%")

p.plot_test_set()