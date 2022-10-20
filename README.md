# AutoTrading-Template
# 標準差 Standard Deviation
>標準差應用於投資上，可作為量度回報穩定性的指標。標準差數值越大，代表回報遠離過去平均數值，回報較不穩定故風險越高。相反，標準差數值越小，代表回報較為穩定，風險亦較小。
Please clearly write your idea.
# Code 簡述
> 用numpy計算traning標準差:
- 以收盤價與開盤價比較
 - 漲幅超過 standard deviation 則賣
 - 位於 <standard deviation和0之間 則不買
 - 跌幅低於 0 則買
-standard deviation = 0.5882725251559258
# Requirments
python : 3.9.13

numpy==1.19.5

pandas==1.1.5

scikit-learn==0.24.1

# Use
python stock_BuySell.py [training_data.csv] [testing_data.csv] [output.csv]
