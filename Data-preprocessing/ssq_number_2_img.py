# 双色球7位中奖号码转化为28x28的图片
import numpy as np
import pandas as pd
import cv2 as cv

df = pd.read_csv('ssq_asc.txt', sep=' ', header=None)  # 假定分隔符为空格
data = df.iloc[:, 2:9]
data_np = data.to_numpy()

i = 0

for row in data_np:
    i = i + 1
    img = np.concatenate((row,row,row,row),axis=0)
    img = np.array((img,img,img,img,img,img,img,
                    img,img,img,img,img,img,img,
                    img,img,img,img,img,img,img,
                    img,img,img,img,img,img,img
                    ), ndmin=2) # 28x28
    
    cv.imwrite('dataset/'+ str(i) +'.jpg', img*7)
    

