# AutoTrading-Template

Please clearly write your idea.
# Code 簡述
- 用numpy計算標準差:
以收盤價與開盤價比較
漲幅超過 delta 則買
跌幅超過 delta 則賣
delta defult = 0.005
第七天後的策略 :
RSI < 25 BUY
RSI > 65 SELL
# Requirments
python : 3.9.13

numpy==1.19.5

pandas==1.1.5

scikit-learn==0.24.1

# Use
python stock_BuySell.py [training_data.csv] [testing_data.csv] [output.csv]
