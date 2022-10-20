# You can write code above the if-main block.
import csv
import numpy as np
import pandas as pd
import sklearn
import statistics

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

FEATURE_NAMES = ('open', 'high', 'low', 'close')
training = pd.read_csv('training_data.csv', names=FEATURE_NAMES)
FEATURE_NAMES = ('open', 'high', 'low', 'close')
testing = pd.read_csv('testing_data.csv', names=FEATURE_NAMES)
training['new']= training['open']-training['close']
tr = statistics.stdev(training['new'])
testing['new']= testing['open']-testing['close']
BUY = 1
NO_ACTION = 0
SOLD = -1
#number of stock
num = 0
sum_price = 0
f=open('output.csv', 'wt', newline='')
for i in range(len(testing)):
    if i < 19:
        if testing['new'][i] < 0:
            if num > 0:
                print (0,file=f)
            else:
                print (1,file=f)
                num+=1
        elif tr > testing['new'][i] > 0:
                print (0,file=f)
        else:
            if num < 0:
                print (-1,file=f)
            else:
                print (0,file=f)
                num-=1    
f.close()
