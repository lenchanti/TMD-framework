import os
import csv
import shutil
import re
import numpy as np
from scipy import interpolate
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from glob import glob
import math
import torch



def DeleteAllfiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return"REMOVE ALL FILE"
    else:
        return "Directory Not Found"
    

def min_max_normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
# z-변환 함수 정의
def z_normalize(data):
    return (data - np.mean(data)) / np.std(data)

def mean_crossing_rate(series):
    mean_val = series.mean()
    # 평균을 기준으로 값이 바뀌는 지점 확인
    crossings = np.diff(np.sign(series - mean_val))
    mcr = np.sum(crossings != 0)
    return mcr

# 각 열에 대한 mean, std, min, max, MCR 계산

path="D:\논문과제1\crc1st_data_1_60\merge"
print(DeleteAllfiles("D:\논문과제1\prep\\five_sec"))
#os.chdir("D:\논문과제1\prep")
#피험자 별 파일 불러 오기
#os.chdir(path)

df_file=pd.DataFrame()
mode=pd.DataFrame()
for sample_files in os.listdir(path):
    sample_file=pd.read_csv(path+"\\"+sample_files)
    print(sample_files)
    sample_mode_file=sample_file[['Mode']]
    col_names=sample_file.iloc[:,1:-2].columns
    df_file=pd.concat([df_file,sample_file],axis=0,ignore_index=True)
    mode=pd.concat([mode,sample_mode_file],axis=0,ignore_index=True)
    
df_file= df_file.apply(z_normalize)    


df=pd.DataFrame()
count=0
for i in range(300,len(df_file)+1,300): 
    count+=1
    result=pd.DataFrame()
    sample=df_file.iloc[i-300:i-1,]
        
        
    sample_cols=col_names
        
         
    mean_val=sample.mean().values.reshape(1, -1)
    mean_val=pd.DataFrame(mean_val,columns="mean_"+sample_cols)
        
    std_val=sample.std().values.reshape(1, -1)
    std_val=pd.DataFrame(std_val,columns="std_"+sample_cols)
        
    result=pd.concat([result,mean_val], axis=1)
    result=pd.concat([result,std_val], axis=1)
        
    mean_crossing_rate_val=sample.apply(mean_crossing_rate).values.reshape(1, -1)
    mean_crossing_rate_val=pd.DataFrame((mean_crossing_rate_val), columns="mcr"+sample_cols)
       
    result=pd.concat([result,mean_crossing_rate_val], axis=1)
        
    min_val=sample.min().values.reshape(1, -1)
    min_val=pd.DataFrame(min_val,columns="min_"+sample_cols)
        
    result=pd.concat([result,min_val], axis=1)
        
    max_val=sample.max().values.reshape(1, -1)
    max_val=pd.DataFrame(max_val,columns="max_"+sample_cols)
        
    result=pd.concat([result,max_val], axis=1)
        
        ####################################
        #FFT 수행 Fast Fourier Transform 사용
    fft_result = np.fft.fft(sample,axis=1)

        # FFT 결과의 크기 계산
    magnitude = np.abs(fft_result)

        # 가장 큰 크기 값 찾기
    magnitude=pd.DataFrame(magnitude,columns="highest_magnitude_"+sample_cols)
        
    highest_magnitude = magnitude.max().values.reshape(1,-1)
    highest_magnitude=pd.DataFrame(highest_magnitude,
                                       columns="highest_mag_"+sample_cols
                                       )

    result=pd.concat([result,highest_magnitude],axis=1)
        
    df=df.reset_index(drop=True)
    result=result.reset_index(drop=True)
       
    df=pd.concat([df,result],axis=0,ignore_index=True)
    print(count)
        
    
    

df_mode=pd.DataFrame()
for j in range(300,len(sample_mode_file)+1,300):
    df_mode=pd.concat([df_mode,sample_mode_file[j-1:j]],
                          ignore_index=True)
        
        
df=df.reset_index(drop=True)
df_mode=df_mode.reset_index(drop=True)
           
df=pd.concat([df,df_mode],axis=1)
        
        
os.chdir("D:\논문과제1\prep_3rd")
print("save")
df.to_csv("extraction_5s_"+sample_files)
   