from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('final_data.csv')
df = pd.DataFrame(data)
df.columns = ['date', 'high_price', 'low_price', 'mid_price', 'prediction_price']

mid_price = df['mid_price']
pred_price = df['prediction_price']
time_stamp = df['date']

plt.figure(figsize=(20, 10))
plt.plot(time_stamp, mid_price, color='red')
plt.plot(pred_price, color='blue')
plt.title('Chart Data')
plt.xlabel('Time stamp (UTC)')
plt.ylabel('mid_price')

plt.show()