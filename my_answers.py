import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# transform the input series and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size - 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# build an RNN to perform regression on our time series input/output data
#    layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
#    layer 2 uses a fully connected module with one unit

def build_part1_RNN(window_size):
    m = Sequential()
    m.add(LSTM(units = window_size, input_shape = (window_size,1)))
    m.add(Dense(1))
    
    return m

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
