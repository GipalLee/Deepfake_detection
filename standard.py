# Standard Normal Distribution
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#s_scaler = MinMaxScaler()
s_scaler = StandardScaler()
# load the filenames for train videos
TRAIN_PATH = './videos/mine/'

df = pd.read_csv('./videos/mine/variance_done.csv')
df = s_scaler.fit_transform(df)
print(df)
df = pd.DataFrame(df)
df.to_csv("./videos/mine/stand-done.csv",index=False,header=True)