import pandas as pd

df = open('chart_data.csv', 'r', encoding='utf-8')

# CSV에 저장한 데이터를 이용해 중간값 추출
high_low = df[['high_price', 'low_price']]
print(high_low)

high_low['mid_price'] = int(high_low.sum(axis = 1)) / 2
print(high_low)

dataframe = df['candle_date_time_utc']
dataframe.assign(mid_price = high_low['mid_price'])


# 최종 데이터를 저장할 CSV 파일
final_csv = open("num_data.csv", "w")
dataframe.to_csv(final_csv, sep = ",", na_rep = 'NaN')