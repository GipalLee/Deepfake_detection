# Standard Normal Distribution
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#s_scaler = MinMaxScaler()
s_scaler = StandardScaler()
# load the filenames for train videos
TRAIN_PATH = './videos/sample/'
train_fns = sorted(glob.glob(TRAIN_PATH + '*.csv'))
print('There are {} samples in the train set.'.format(len(train_fns)))

resfile = open("var.csv", "a")
resfile.write(
    "mse,psnr,ssim,hist_diff,r_avg_diff,g_avg_diff,b_avg_diff,r_max_diff,g_max_diff,b_max_diff,h_avg_diff,s_avg_diff,v_avg_diff,h_max_diff,s_max_diff,v_max_diff,matrix_diff_r,matrix_diff_g,matrix_diff_b, totalentropy, totalvariance, edgedensity, edgeentropy, dctcoefficient,deepfake\n")

for file in train_fns :
    fake = 0
    if file.startswith('./videos/sample\\fake'):
        fake = 1
    df = pd.read_csv(file)
    for i in range(len(df.columns)):
        print(i)
        df.iloc[:, i] = df.iloc[:, 0] / df.iloc[:, i].mean()


    for i in range(0,len(df),30) :
        cur = df[i:i+30].var(axis=0)

        resfile.write(
            str(cur[0]) + ',' + str(cur[1]) + ',' + str(cur[2]) + ',' + str(cur[3]) + ',' +
            str(cur[4]) + ',' + str(cur[5]) + ',' + str(cur[6]) + ',' +
            str(cur[7]) + ',' + str(cur[8]) + ',' + str(cur[9]) + ',' +
            str(cur[10]) + ',' + str(cur[11]) + ',' + str(cur[12]) + ',' +
            str(cur[13]) + ',' + str(cur[14]) + ',' + str(cur[15]) + ',' +
            str(cur[16]) + ',' + str(cur[17]) + ',' + str(cur[18]) + ',' +
            str(cur[19]) + ',' + str(cur[20]) + ',' + str(cur[21]) + ',' +
            str(cur[22]) + ',' + str(cur[23]) + ',' + str(fake) + "\n"
        )

#df = pd.read_csv('./1.csv')
#df = s_scaler.fit_transform(df)