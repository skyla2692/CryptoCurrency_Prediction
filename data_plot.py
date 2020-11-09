from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('num_data.csv')
df = pd.DataFrame(data)
df.columns = ['candle_date_time_utc', 'high_price', 'low_price', 'mid_price']

mid_price = df['mid_price']
time_stamp = df['candle_date_time_utc']

plt.plot(time_stamp, mid_price, color='purple')
plt.title('Chart Data')
plt.xlabel('Time stamp (UTC)')
plt.ylabel('mid_price')

plt.show()