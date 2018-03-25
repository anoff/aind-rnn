import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# transform the input series and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size):
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
    m.add(LSTM(5, input_shape = (window_size,1)))
    m.add(Dense(1))
    
    return m

### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    abc = list(string.ascii_lowercase)
    text = [c for c in list(text) if c in punctuation + abc]

    return "".join(text)

### transform the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])

    return inputs,outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
#layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
#layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
#layer 3 should be a softmax activation ( since we are solving a multiclass classification)
def build_part2_RNN(window_size, num_chars):
    m = Sequential()
    m.add(LSTM(200, input_shape=(window_size, num_chars)))
    m.add(Dense(num_chars, activation='softmax'))
    return m
