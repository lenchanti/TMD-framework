import os
import shutil
import re
import numpy as np
from scipy import interpolate
import pandas as pd
#import tensorflow_decision_forests as tfdf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from glob import glob

#import tensorflow as tf
#from tensorflow import keras

#Python, Keras, TensorFlow, Pandas, and numpy


def DeleteAllfiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return"REMOVE ALL FILE"
    else:
        return "Directory Not Found"
    
    
  
#데이터 읽어오기
path="D:\논문과제1\crc3rd_data"
# path="D:\논문과제1\crc1st_data_60_119"
print(DeleteAllfiles("D:\논문과제1\crc3rd_data_merge"))
#print(DeleteAllfiles(path+"\\merge"))
print(os.listdir(path))

files_list=[]
timeline=pd.DataFrame({"Time":[]})
#60HZ 주파수로 resampling
count=0
for j in range(30,630):
    for i in range(1,61):
        count+=1
        
        timeline.loc[count-1]=(j+round(i*0.0166,4))
        print(count)
        
lst=[]
num=0
for folders in os.listdir(path):
        
      if ("no_error") in folders:
          if "" in folders:
              print(folders)
              _path=path
              _path=path+"\\"+folders
              merge_df=pd.DataFrame()
              file_num = 0
              file_count = 0
              for file_path in os.listdir(_path):
                  file_num = len((os.listdir(_path)))
                  file_count += 1
              
                  if 'SensorData' in file_path:
             
                      file_df=pd.read_csv(_path+"\\"+file_path,low_memory=False)
                      idf=file_df.iloc[:,-2:] 
                      idf=pd.concat([idf["Mode"].fillna(idf["Mode"][0]),
                          idf["Survay1"].fillna(idf["Survay1"][0])] , axis=1)
                      idf=idf.iloc[:36000]
                      # print(set(idf["Mode"]))
             
              
                
                      file_df=pd.concat([file_df.iloc[:,:-2],idf],axis=1)
                
                      file_concat_df=pd.concat([file_df.iloc[:,:-2],timeline],ignore_index=True)
               
                
                      file_concat_df=file_concat_df.sort_values(by="Time")
                      file_concat_df=file_concat_df.interpolate()
                      file_concat_df=file_concat_df.reset_index()
                      file_concat_df=file_concat_df.sort_values(by="index")
                 
                      file_concat_df=file_concat_df.set_index(["index"])
                
                      df=file_concat_df[len(file_df):len(file_concat_df)]
                      df=df.interpolate()
                      df=df.reset_index(drop=True)
                      df = df[['GraX', 'GraY', 'GraZ',
                     'LinearAccX', 'LinearAccY', 'LinearAccZ', 
                     'GyroX', 'GyroY', 'GyroZ',
                     'MagX', 'MagY', 'MagZ'
                     ]]
                      df=pd.concat([df,idf],axis=1)
                
               
                      df=df.reset_index(drop=True)
                      merge_df=pd.concat([merge_df,df],axis=0,ignore_index=True)
                      
                      a=len(set(df["Mode"]))
                      df=df.dropna(subset=["Mode"])
                      b=len(set(df["Mode"]))
                      if(a!=b):
                          print(file_path)
                    
                #print(set(merge_df["Mode"]))
                
                
                  if (file_count==(file_num)):
                      os.chdir("D:\논문과제1\crc3rd_data_merge")
                      merge_df.to_csv(folders.replace("#", "")+"_merge.csv",index=False)
                    
                    
    
import pandas as pd

er_df=pd.read_csv("D:\논문과제1\crc3rd_data\\2021_1_30_18_0_49_still_SensorData.csv",low_memory=False)
er_df_idf=pd.DataFrame()
# s.head()
er_df_idf=er_df_idf.reindex(index=range(36000),columns=["Mode"])

er_df_idf["Mode"][0]=(er_df["Mode"][0])

er_df_idf=pd.DataFrame(er_df_idf["Mode"].fillna(er_df_idf["Mode"][0]),columns=["Mode"])
        
    # print(set(idf["Mode"]))

files_list=[]
timeline=pd.DataFrame({"Time":[]})
#60HZ 주파수로 resampling
count=0
for j in range(30,630):
    for i in range(1,61):
        count+=1
        timeline.loc[count-1]=(j+round(i*0.0166,4))
        print(count)
                      
er_df=er_df[["Time",'GraX', 'GraY', 'GraZ',
                      'LinearAccX', 'LinearAccY', 'LinearAccZ', 
                      'GyroX', 'GyroY', 'GyroZ',
                      'MagX', 'MagY', 'MagZ'
                      ]]              
er_df_c=pd.concat([er_df,timeline],ignore_index=True)
               
er_df_c=er_df_c.sort_values(by="Time")
er_df_c=er_df_c.interpolate()
er_df_c=er_df_c.reset_index()
er_df_c=er_df_c.sort_values(by="index")
                 
er_df_c=er_df_c.set_index(["index"])
                
er_df=er_df_c[len(er_df):len(er_df_c)]

df=pd.DataFrame()
er_df=er_df.reset_index(drop=True)
er_df_idf=er_df_idf.reset_index(drop=True)
df=pd.concat([er_df,er_df_idf],axis=1)

df=df.iloc[:,1:]

os.chdir("D:\논문과제1\crc3rd_data_merge")
df.to_csv("er_2021_1_30_18_0_49_still_SensorData.csv",index=False)
    
    
    

