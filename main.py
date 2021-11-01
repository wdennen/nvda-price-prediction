# machine learning project 3
import numpy
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'pink', 'brown']
LEGEND_LABELS = ['Training', 'Validation', 'Testing']
ACTIVATIONS = ['sigmoid', 'softmax', 'tanh', 'selu', 'relu', 'gelu']

# create preprocessing MinMax scaler to normalize points between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))


def extract(file, days, preprocess, part):
    # extract each adjusted closing rate from the rows of the csv
    adj_closings = []
    for i, line in enumerate(file):
        if i:
            adj_closings.append(float(line.strip().split(',')[5]))
    n = len(adj_closings)
    # normalize data if preprocess is requested and doing lstm
    if preprocess and part == 'lstm':
        adj_closings = scaler.fit_transform(numpy.array(adj_closings).reshape(-1, 1))
    samples = []
    labels = []

    for i in range(days, n):
        samples.append(adj_closings[i - days:i])
        labels.append(adj_closings[i])
    return samples, labels


def create_model_ann(days, neurons, layers):
    model = Sequential()
    act = ACTIVATIONS[4]
    if layers == 2:  # 1 hidden layer
        model.add(Dense(neurons, input_dim=days, activation='relu'))
    elif layers == 3:  # 2 hidden layers
        model.add(Dense(neurons, input_dim=days, activation=ACTIVATIONS[5]))
        model.add(Dense(neurons, activation=ACTIVATIONS[5]))
    else:  # 3 hidden layers
        model.add(Dense(neurons, input_dim=days, activation=act))
        model.add(Dense(neurons, activation=act))
        model.add(Dense(neurons, activation=act))

    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model


def train_model_ann(model, training_samples, training_labels, neurons):
    history = model.fit(training_samples, training_labels, epochs=150, batch_size=10, validation_split=0.2)

    plt.figure(1)
    plt.plot(history.history['val_loss'], label='Validation')
    plt.plot(history.history['loss'], label='Training', color='red')
    plt.title('Training/Validation Accuracy vs. Epoch (' + str(neurons) + ' neurons)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    return model


def test_model_ann(model, test_samples, test_labels, neurons):
    y_true = test_labels
    y_pred = model.predict(test_samples)
    mse = mean_squared_error(y_true, y_pred)

    plt.figure(2)
    plt.plot(y_pred, label='Predicted Value', color='red')
    plt.plot(y_true, label='Actual Value', linestyle=(0, (1, 5)))

    plt.title('Predicted vs. Actual Value (' + str(neurons) + ' neurons) -- Average MSE: ' + str(round(mse)))
    plt.xlabel('Days Since 10/1/16')
    plt.ylabel('Adjusted Closing Price')
    plt.legend()

    return mse


def create_model_lstm(blocks, days):
    model = Sequential()
    # model.add(LSTM(blocks, return_sequences=True, input_shape=(1, days)))
    model.add(LSTM(blocks, input_shape=(1, days)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model_lstm(model, training_samples, training_labels):
    model.fit(training_samples, training_labels, epochs=20, batch_size=10, validation_split=0.2)
    return model


def test_model_lstm(model, blocks, test_samples, test_labels, preprocess):
    y_true = numpy.array(test_labels)
    y_pred = model.predict(test_samples)

    # invert results and test data to return to original units if data is normalized, and calculate MSE
    if preprocess:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        mse = mean_squared_error(y_true[0], y_pred[0])
    else:
        mse = mean_squared_error(y_true, y_pred[:, 0])

    plt.figure(1)
    plt.plot(y_pred, label='Predicted Value', color='red')
    plt.plot(y_true, label='Actual Value', linestyle=(0, (1, 5)))

    plt.title('Predicted vs. Actual Value (' + str(blocks) + ' blocks) -- MSE: ' + str(round(mse, 2)))
    plt.xlabel('Days Since 10/1/16')
    plt.ylabel('Adjusted Closing Price')
    plt.legend()
    return mse


def main():
    file = open('NVDA.csv')

    part = 'ann'
    preprocess = True

    # extract adjusted closing prices from data
    days = 10  # past days per data label, either 7 or 10
    samples, labels = extract(file, days, preprocess, part)
    n = len(samples)

    # split data into training and validation (first 80%) and testing (final 20%)
    training_samples, training_labels = numpy.array(samples[days:n - round(0.2 * n)]), numpy.array(
        labels[days:n - round(0.2 * n)])
    test_samples, test_labels = samples[n - round(0.2 * n):n], labels[n - round(0.2 * n):n]

    if part == 'ann':
        # specify model parameters
        layers = 3  # total amount of layers (including output layer)
        neurons = 30  # neurons per hidden layer

        # create, train, and test model
        model = create_model_ann(days, neurons, layers)
        model = train_model_ann(model, training_samples, training_labels, neurons)
        mse = test_model_ann(model, test_samples, test_labels, neurons)

        # display results
        print(mse)
        plt.show()

    if part == 'lstm':
        # reshape/reformat data input to properly fit LSTM model
        test_samples = numpy.array(test_samples)
        training_samples = numpy.reshape(training_samples, (training_samples.shape[0], 1, training_samples.shape[1]))
        test_samples = numpy.reshape(test_samples, (test_samples.shape[0], 1, test_samples.shape[1]))

        # specify model parameters
        blocks = 128

        # create, train, and test model
        model = create_model_lstm(blocks, days)
        model = train_model_lstm(model, training_samples, training_labels)
        mse = test_model_lstm(model, blocks, test_samples, test_labels, preprocess)

        # display results
        print(mse)
        plt.show()

    file.close()


if __name__ == '__main__':
    main()
