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
    print(data)
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
        tmp = data_pad[i:i+ksize] * kernel
        for j in range(data.shape[1]):
            data.iloc[i,j] = np.sum(tmp[:,j])
    print(data)
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

def makeDecision(state, balance, dp1, dp2) -> int:
    decision = 0
    if dp1[0] < 0:
        if state == 1:
            decision = 0
        elif state == 0:
            decision = 1
        else :
            decision = 1
    elif dp1[0] > 0:
        if state == 1:
            decision = -1
        elif state == 0:
            decision = -1
        else :
            decision = 0
    return decision


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    BATCH_SIZE = 32
    REF_DAY = 30
    PREDICT_DAY = 3
    TRAIN_RATIO = 0.8
    EPOCH = 120
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

    # save model
    save_name = '{}_{}.h5'.format(int(time.time()), np.around(np.min(history.history['val_mse']), decimals=4))
    lstm_model.save(save_name)

    # output predict option
    testing_data_raw = pd.read_csv(args.testing, header=None, usecols=[0,1,2,3], names=['open', 'high', 'low', 'close'])
    testing_data = testing_data_raw.apply(lambda x: (x - data_min) / data_diff)

    tmp = []
    predict_input_raw = np.array(data_norm.iloc[-REF_DAY:])
    tmp.append(predict_input_raw)
    predict_input = np.array(tmp)
    predict_output = lstm_model.predict(predict_input)[0]
    predict_output = recoverNormalize(predict_output, data_min, data_diff)
    
    predict_res = []
    predict_res.append(predict_output[0])
    cache = predict_output[0]

    state = 0
    dp1 = [0, 0]
    dp2 = 0
    hold_price = 0
    print('hold price : {}'.format(hold_price))

    with open(args.output, "w") as output_file:
        for write_cnt in range(testing_data.shape[0] - 1):
            predict_input_raw = np.vstack((predict_input_raw, testing_data.iloc[write_cnt]))[1:]

            tmp = []
            tmp.append(predict_input_raw)
            predict_input = np.array(tmp)
            predict_output = lstm_model.predict(predict_input)[0]
            predict_output = recoverNormalize(predict_output, data_min, data_diff)
            predict_res.append(predict_output)
            print(predict_output)

            # condition val 
            tmp_price = testing_data_raw.iloc[write_cnt, 0]
            balance = tmp_price - hold_price
            dp1 = [predict_output[0] - cache, predict_output[1] - predict_output[0]]
            dp2 = dp1[1] - dp1[0]
            cache = predict_output[0]

            # make decision
            decision = makeDecision(state, balance, dp1, dp2)
            old_state = state
            state += decision
            if (old_state == 1 and state == 0) or (old_state == -1 and state == 0):
                hold_price = 0
            elif (old_state == 0 and state == 1) or (old_state == 0 and state == -1):
                hold_price = tmp_price
            assert state <= 1 or state >= -1, 'something wrong!!!'
            print('decision : {}'.format(decision))

            output_file.write('{}\n'.format(decision))


        fig, ax = plt.subplots(figsize=(20, 10))
        for i in range(len(predict_res)):
            ax.plot(predict_res[i], label='predict')  
        ax.set_xlabel('index')
        ax.set_ylabel('open price')
        ax.set_title('Predict Result')
        ax.legend()
        plt.show()
        plt.savefig('predict.png')

        predict_res = np.array(predict_res)
        print(predict_res)