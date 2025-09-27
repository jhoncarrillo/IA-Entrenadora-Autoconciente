from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore

def create_rnn(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
