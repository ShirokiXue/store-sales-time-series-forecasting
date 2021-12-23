# Computational imports
import numpy as np   # Library for n-dimensional arrays
import pandas as pd  # Library for dataframes (structured data)

# Helper imports
import os 
import warnings
# import pandas_datareader as web
import datetime as dt

# ML/DL imports
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_dataset(family='AUTOMOTIVE', store_nbr=1):

    filt = full_dataset['family'] == family
    dataset = full_dataset.loc[filt]
    filt2 = dataset['store_nbr'] == store_nbr
    dataset = dataset.loc[filt2]

    return dataset

def Multi_Step_LSTM_model(history_input, num_feature_input):
    
    # Use Keras sequential model
    model = Sequential()    
    
    # First LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    model.add(LSTM(units = 64, activation='relu', input_shape = (history_input, num_feature_input))) 
    model.add(Dropout(0.2))
    
    # # Second LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    # model.add(LSTM(units = 16,  activation='relu'))                                   
    # model.add(Dropout(0.2))
    
    
    # The output layer with linear activation to predict Open stock price
    model.add(Dense(units=1, activation = "linear"))
    
    return model

path = 'input/'

train_data = pd.read_csv(path+'train.csv', index_col=0)
test_data = pd.read_csv(path+'test.csv', index_col=0)
full_dataset = pd.concat([train_data, test_data], ignore_index=True, sort=False)

family = []
for f in full_dataset['family']:
    if f not in family:
        family.append(f)

store_nbr = []
for nbr in full_dataset['store_nbr']:
    if nbr not in store_nbr:
        store_nbr.append(nbr)

for f in family:
    for nbr in store_nbr:

        dataset = get_dataset(f, nbr)
        scaler = []

        for i, col in enumerate(['sales', 'onpromotion']):
            scaler.append(MinMaxScaler(feature_range=(0,1)))
            null_index = dataset[col].isnull()
            dataset.loc[~null_index, [col]] = scaler[i].fit_transform(dataset.loc[~null_index, [col]])

        x_train = dataset[['sales', 'onpromotion']][:-16].to_numpy()
        y_train = dataset[:-16].sales.to_numpy()

        num_feature_input = 2
        history_input = 15
        generator = TimeseriesGenerator(x_train, y_train, length=history_input, batch_size = 1)
        
        try:
            model = load_model(f'output\model\model_{f}_{nbr}.h5')
            print(f"load model_{f}_{nbr}.h5")
        except OSError:
            model = Multi_Step_LSTM_model(history_input, num_feature_input)
            model.compile(optimizer='adam', loss='mean_squared_error')
            print(f"training {f}_{nbr}...")
            model.fit(generator, steps_per_epoch=len(generator), epochs=10, verbose=1)
            model.save(f'output\model_{f}_{nbr}.h5')

        test = dataset[1700-16-history_input:].copy()
        for i in range(16):
            id = test.index[history_input+i]
            X = np.array([test[['sales', 'onpromotion']][0+i:history_input+i].copy().to_numpy()])
            y = model.predict(X)
            test.loc[id, 'sales'] = y[0]
            y = scaler[0].inverse_transform(y)
            full_dataset.loc[id, 'sales'] = y[0][0]

        submission = pd.concat([pd.Series(range(len(full_dataset)) ,name = "id"),   full_dataset.sales], axis = 1)
        submission = submission.loc[3000888:]
        submission.to_csv('output/submission.csv',index=False)