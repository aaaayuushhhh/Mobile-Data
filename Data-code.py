import opendatasets as od
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

od.download('https://www.kaggle.com/iabhishekofficial/mobile-price-classification')

st.title("Mobile Phone Data")
dt = pd.read_csv("mobile-price-classification/train.csv")
df = pd.read_csv("mobile-price-classification/test.csv")

# print(dt.describe())


# pie
st.subheader("Shows the pie graph for number of mobiles using Bluetooth")
fig1 = plt.figure(figsize=(12 , 7))
gg = dict(df.groupby('blue')['blue'].count())
plt.pie(gg.values(), labels=gg.keys(), autopct='%1.0f%%')
st.pyplot(fig1)



# mean
st.subheader("Shows Mean of Batteries used by multiple smartphones")
# Battery_mean = dict(dt.groupby('battery_power')['battery_power'].count())
Battery_mean = list(dt['battery_power'])
st.write(np.mean(list(Battery_mean)))



#bar-graph for battery and cam
st.subheader("Shows Bar graph for battery and camera stats in smartphones")
fig2 = plt.figure(figsize=(12 , 7))
plt.bar(dt.battery_power, dt.fc)
plt.xlabel('Battery(Mah)')
plt.ylabel('Camera(MP)')
st.pyplot(fig2)


#scatter-graph for Ram and Battery
st.subheader("Scatter graph between RAM and BATTERY")
fig3 = plt.figure(figsize=(12 , 7))
plt.scatter(dt.ram, dt.fc)
plt.xlabel('Ram(MB)')
plt.ylabel('Camera(Mpx)')
st.pyplot(fig3)


#heatmap for battery and talk-time
st.subheader("Heat-Map for Ram , Weight , Core Speed and Pixels of a Mobile")
fig2 = plt.figure(figsize=(12 , 8))
sns.heatmap(
    dt.loc[:, ['ram', 'mobile_wt', 'clock_speed', 'px_width']].corr(),
    annot=True
)
st.pyplot(fig2)



# st.subheader("Scatter graph between Battery and talktime")
# fig4 = plt.figure(figsize=(12, 7))
# plt.bar(dt.battery_power, dt.talk_time)
# plt.xlabel('Battery(Mah)')
# plt.ylabel('Talktime(Hours)')
# st.pyplot(fig4)
