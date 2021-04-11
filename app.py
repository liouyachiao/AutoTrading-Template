# You can write code above the if-main block.
from numpy.lib.npyio import save
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# normalize data to [0,1]
def normalize(data):
    data_norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data_norm

def recoverNormalize(data, min_, diff_):
    return np.round(data[:] * diff_ + min_, decimals=2)

def gaussianBlur(data, ksize=3):
    assert ksize > 0 and ksize % 2 == 1, "ksize must be positive odd integer"

    bias = (ksize - 1) // 2
    
    x = []
    for i in range(ksize):
        x.append([-bias + i])
    x = np.array(x)
    x = np.exp(-x**2)
    kernel = x / x.sum()
    pb, pe = [], []
    # padding begin
    for i in range(bias):
        pb.append(data.iloc[0].values)
    # padding end
    for i in range(bias):
        pe.append(data.iloc[-1].values)
    pb, pe = np.array(pb), np.array(pe)
    data_pad = np.vstack((pb, data.values, pe))
    for i in range(data.shape[0]):
        # tmp = data_pad[i:i+ksize] * kernel
        tmp = data_pad[i:i+ksize]
        for j in range(data.shape[1]):
            data.iloc[i,j] = np.sum(tmp[:,j]) / ksize
    
    return data

# use the number of ref_day's data to predict the number of predict_day's data
# split training/testing data by 'ratio' parameter
def setTrainTestData(data, ref_day, predict_day):
    x, y = [], []
    for i in range(len(data) - ref_day - predict_day):
        x.append(np.array(data.iloc[i:i + ref_day]))
        y.append(np.array(data.iloc[i + ref_day:i + ref_day + predict_day]['open']))
    x,y = np.array(x), np.array(y)
    return x, y

# use random to reorder data
def shuffle(x, y):
    np.random.seed(int(time.time()))
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

# use LSTM model as training model
def buildManyToManyModel(in_shape, out_shape, batch_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(in_shape[1], in_shape[2]), return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    # # Adding a second LSTM layer and some Dropout regularisation
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(TimeDistributed(Dense(units = 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(out_shape))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    model.summary()
    return model

# plot the loss, mse diagram and save it
def lossDump(history):
    fig, ax = plt.subplots()
    ax.plot(history['val_loss'], label='val_loss')
    ax.plot(history['val_mse'], label='val_mse')
    ax.plot(history['loss'], label='loss')
    ax.plot(history['mse'], label='mse')
    ax.set_ylabel('result')
    ax.set_xlabel('epoch')
    ax.set_title('history')
    ax.legend()
    plt.savefig('loss.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    BATCH_SIZE = 32
    REF_DAY = 30
    PREDICT_DAY = 2 
    TRAIN_RATIO = 0.8
    EPOCH = 100
    PATIENCE = 20
    KERNEL_SIZE = 5

    # load data 
    df = pd.read_csv(args.training, header=None, usecols=[0,1,2,3], names=['open', 'high', 'low', 'close'])
    data_raw = df

    fig, ax = plt.subplots(2, figsize=(20, 10))
    ax[0].plot(data_raw['open'], label='raw')
    ax[0].set_title('Raw Data')
    ax[0].set_xlabel('index')
    ax[0].set_ylabel('open price')
    ax[0].legend()

    data_conv = gaussianBlur(data_raw, KERNEL_SIZE)

    data_min, data_max = np.min(data_conv['open'].values), np.max(data_conv['open'].values)
    data_diff = data_max - data_min 
    print("max : {}, min : {}, diff : {}".format(data_max, data_min, data_diff))

    data_norm = normalize(data_conv)

    ax[1].set_title('Data After Gaussian Blur (Kernel Size = {})'.format(KERNEL_SIZE))
    ax[1].plot(data_conv['open'], label='blur')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('open price')
    ax[1].legend()
    plt.savefig('GaussianBlur.png')
    plt.close()


    x_raw, y_raw = setTrainTestData(data_norm, REF_DAY, PREDICT_DAY)

    x_train, y_train = shuffle(x_raw[:int(x_raw.shape[0] * TRAIN_RATIO)], y_raw[:int(y_raw.shape[0] * TRAIN_RATIO)])
    x_test, y_test = shuffle(x_raw[int(x_raw.shape[0] * TRAIN_RATIO):], y_raw[int(y_raw.shape[0] * TRAIN_RATIO):])
    
    lstm_model = buildManyToManyModel(x_train.shape, PREDICT_DAY, BATCH_SIZE)
    early_stopping = EarlyStopping(monitor='val_mse', patience=PATIENCE, verbose=1, mode='min')
    # filepath="weights-{epoch:02d}-{val_mse:.4f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
    history = lstm_model.fit(x_train, y_train, verbose=1, callbacks=[early_stopping],\
        validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCH)

    lossDump(history.history)

    predict_input = []
    predict_input.append(np.array(data_norm.iloc[-REF_DAY:]))
    predict_input = np.array(predict_input)

    predict_output = lstm_model.predict(predict_input)[0]
    predict_output = recoverNormalize(predict_output, data_min, data_diff)
    print(predict_output)

    # save model
    save_name = '{}_{}.h5'.format(int(time.time()), np.around(np.min(history.history['val_mse']), decimals=4))
    lstm_model.save(save_name)



    # trader = Trader()
    # trader.train(training_data)

    # testing_data = load_data(args.testing)
    # with open(args.output, "w") as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)

    #         # this is your option, you can leave it empty.
    #         trader.re_training()
