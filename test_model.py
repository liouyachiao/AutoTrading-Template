from numpy.lib.npyio import save
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training", default="training.csv", help="input training data file name")
parser.add_argument("--testing", default="testing.csv", help="input testing data file name")
parser.add_argument("--output", default="test.csv", help="output file name")
parser.add_argument("--model", default="result.h5", help="output file name")
args = parser.parse_args()

REF_DAY = 30
PREDICT_DAY = 1

model = keras.models.load_model(args.model)

# input train
df = pd.read_csv(args.training, header=None, usecols=[0, 1, 2, 3], names=['open', 'high', 'low', 'close'])
df_min, df_max = np.min(df['open'].values), np.max(df['open'].values)
df_diff = df_max - df_min 
df_norm = df.apply(lambda x: (x - df_min) / df_diff)

input_raw = df_norm.values[-REF_DAY:]

# input test  
df_test = pd.read_csv(args.testing, header=None, usecols=[0, 1, 2, 3], names=['open', 'high', 'low', 'close'])
df_norm = df_test.apply(lambda x: (x - df_min) / df_diff).values

def makeDecision(state, dp1) -> int:
    decision = 0
    if dp1 < 0:
        if state == 1:
            decision = 0
        elif state == 0:
            decision = 1
        else :
            decision = 1
    elif dp1 > 0:
        if state == 1:
            decision = -1
        elif state == 0:
            decision = -1
        else :
            decision = 0
    return decision

predict = []
predict_1 = []

tmp = []
tmp.append(input_raw)
input_data = np.array(tmp)
output_raw = model.predict(input_data)[0]
output_data = output_raw[:] * df_diff + df_min 
predict_1.append(output_data[0])
cache = output_data[0]

state = 0
dp1 = 0

with open(args.output, "w") as output_file:
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
        dp1 = output_data[0] - cache
        cache = output_data[0]

        # print('tmp = {}, predict1 = {}, predict2 = {}, dp1[0] = {}, dp1[1] = {}'.format(tmp_price, output_data[0], output_data[1], dp1[0], dp1[1]))

        # make decision
        decision = makeDecision(state, dp1)
        state += decision
        assert state <= 1 or state >= -1, 'something wrong!!!'
        print('{}\n'.format(decision))
        output_file.write('{}\n'.format(decision))

        predict_1.append(output_data[0])

fig, ax = plt.subplots(figsize = (12, 6))
ax.plot(df_test['open'], label = 'ans')
ax.plot(predict_1, label = 'predict 1')
ax.legend()
ax.set_xlabel('index')
ax.set_ylabel('open price')
ax.set_title('predict result')
plt.savefig('test.png')

print('process completed')

