from numpy.lib.npyio import save
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import keras

REF_DAY = 30
PREDICT_DAY = 2 

filename = 'lab02/2_01.h5'
model = keras.models.load_model(filename)

# input train
df = pd.read_csv('training.csv', header=None, usecols=[0, 1, 2, 3], names=['open', 'high', 'low', 'close'])
df_min, df_max = np.min(df['open'].values), np.max(df['open'].values)
df_diff = df_max - df_min 
df_norm = df.apply(lambda x: (x - df_min) / df_diff)

input_raw = df_norm.values[-REF_DAY:]

# input test  
df_test = pd.read_csv('testing.csv', header=None, usecols=[0, 1, 2, 3], names=['open', 'high', 'low', 'close'])
df_norm = df_test.apply(lambda x: (x - df_min) / df_diff).values

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

predict = []
predict_1 = []
predict_2 = [150]
predict.append([150, 150, 150])

tmp = []
tmp.append(input_raw)
input_data = np.array(tmp)
output_raw = model.predict(input_data)[0]
output_data = output_raw[:] * df_diff + df_min 
predict_1.append(output_data[0])
predict_2.append(output_data[1])
cache = output_data[0]

state = 0
dp1 = [0, 0]
dp2 = 0
hold_price = 0

with open('test.csv', "w") as output_file:
    for i in range(df_test.shape[0] - 1):

        input_raw = np.vstack((input_raw, df_norm[i]))[1:]

        tmp = []
        tmp.append(input_raw)
        input_data = np.array(tmp)
        output_raw = model.predict(input_data)[0]

        input_data = input_data[:] * df_diff + df_min
        output_data = output_raw[:] * df_diff + df_min 
        print(output_data)
        predict.append(output_data)

        tmp_price = df_test.iloc[i, 0]
        balance = tmp_price - hold_price
        dp1 = [output_data[0] - cache, output_data[1] - output_data[0]]
        dp2 = dp1[1] - dp1[0]
        cache = output_data[0]

        print('tmp = {}, predict1 = {}, predict2 = {}, dp1[0] = {}, dp1[1] = {}'.format(tmp_price, output_data[0], output_data[1], dp1[0], dp1[1]))

        # make decision
        decision = makeDecision(state, balance, dp1, dp2)
        old_state = state
        state += decision
        if (old_state == 1 and state == 0) or (old_state == -1 and state == 0):
            hold_price = 0
        elif (old_state == 0 and state == 1) or (old_state == 0 and state == -1):
            hold_price = tmp_price
        assert state <= 1 or state >= -1, 'something wrong!!!'
        print('{}\n'.format(decision))
        output_file.write('{}\n'.format(decision))

        predict_1.append(output_data[0])
        predict_2.append(output_data[1])

fig, ax = plt.subplots(figsize = (12, 6))
ax.plot(df_test['open'], label = 'ans')
ax.plot(predict_1, label = 'predict 1')
ax.plot(predict_2, label = 'predict 2')
ax.legend()
ax.set_xlabel('index')
ax.set_ylabel('open price')
ax.set_title('predict result')
plt.savefig('test.png')

print('process completed')

